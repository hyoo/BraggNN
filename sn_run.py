import argparse
import time
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from sambaflow import samba
from sambaflow.samba.utils.argparser import parse_app_args
from sambaflow.samba.utils.common import common_app_driver
import sambaflow.samba.utils as utils
# import sambaflow.samba.nn as nn

# from model import model_init, BraggNN
from sn_model import model_init, BraggNN
from util import str2bool, s2ituple
from data import bkgdGen, gen_train_batch_bg, get1batch4test


def add_args(parser: argparse.ArgumentParser) -> None:
  # original args
  # parser.add_argument('-gpus',   type=str, default="", help='list of visiable GPUs')
  parser.add_argument('-expName',type=str, default="debug", help='Experiment name')
  parser.add_argument('-lr',     type=float,default=3e-4, help='learning rate')
  parser.add_argument('-mbsize', type=int, default=512, help='mini batch size')
  parser.add_argument('-maxep',  type=int, default=100000, help='max training epoches')
  parser.add_argument('-fcsz',  type=s2ituple, default='16_8_4_2', help='dense layers')
  parser.add_argument('-psz', type=int, default=11, help='working patch size')
  parser.add_argument('-aug', type=int, default=1, help='augmentation size')
  parser.add_argument('-print',  type=str2bool, default=False, help='1:print to terminal; 0: redirect to file')


def add_run_args(parser: argparse.ArgumentParser) -> None:
  # runtime args
  pass


def train(args: argparse.Namespace, model: nn.Module, optimizer: samba.optim.SGD) -> None:
  mb_data_iter = bkgdGen(
    data_generator=gen_train_batch_bg(mb_size=args.mbsize,
                                      psz=args.psz,
                                      dev=None,
                                      rnd_shift=args.aug),
    max_prefetch=args.mbsize * 4
  )

  # criterion = nn.MSELoss()

  for epoch in range(args.maxep+1):
    time_it_st = time.time()
    X_mb, y_mb = mb_data_iter.next()
    time_data = 1000 * (time.time() - time_it_st)
  
    X_mb = samba.from_torch(X_mb, name='image', batch_dim=0)
    y_mb = samba.from_torch(y_mb, name='label', batch_dim=0)
    hyperparam_dict = {"lr": args.lr}
    loss, pred = samba.session.run(input_tensors=[X_mb, y_mb],
                                   output_tensors=model.output_tensors,
                                   hyperparam_dict=hyperparam_dict,
                                   data_parallel=args.data_parallel,
                                   reduce_on_rdu=args.reduce_on_rdu)
    pred = samba.to_torch(pred)
    # loss = samba.to_torch(loss)

    # outputs = samba.session.run(input_tensors=[X_mb],
    #                             output_tensors=model.output_tensors,
    #                             hyperparam_dict=hyperparam_dict,
    #                             data_parallel=args.data_parallel,
    #                             reduce_on_rdu=args.reduce_on_rdu)
    # pred = outputs[0]
    # loss = criterion(pred, y_mb)
    # loss = loss.torch()

    time_e2e = 1000 * (time.time() - time_it_st)

    if epoch % 2000 == 0:
      if epoch == 0: 
        X_mb_val, y_mb_val = get1batch4test(psz=args.psz, mb_size=6144, rnd_shift=0, dev=None)
        _y_mb_val = torch.from_numpy(y_mb_val)

      samba.session.to_cpu(model)
      with torch.no_grad():
        _, pred_val = model(X_mb_val, _y_mb_val)
           
        pred_train = pred.numpy()
        true_train = y_mb.torch().numpy()
        pred_val = pred_val.torch().numpy()
        l2norm_train = np.sqrt((true_train[:,0] - pred_train[:,0])**2   + (true_train[:,1] - pred_train[:,1])**2) * args.psz
        l2norm_val   = np.sqrt((y_mb_val[:,0] - pred_val[:,0])**2 + (y_mb_val[:,1] - pred_val[:,1])**2) * args.psz

        print('[Train] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f (pixels). time_data: %.2fms, time_e2e: %.2fms' % (\
              (epoch, l2norm_train.shape[0], l2norm_train.mean()) + tuple(np.percentile(l2norm_train, (50, 75, 95, 99.5))) + (time_data, time_e2e) ) )

        print('[Valid] @ %05d l2-norm of %5d samples: Avg.: %.4f, 50th: %.3f, 75th: %.3f, 95th: %.3f, 99.5th: %.3f (pixels) \n' % (\
              (epoch, l2norm_val.shape[0], l2norm_val.mean()) + tuple(np.percentile(l2norm_val, (50, 75, 95, 99.5))) ) )
        
        torch.save(model.state_dict(), "%s/mdl-it%05d.pth" % ('out', epoch))


def get_inputs(args: argparse.Namespace) -> Tuple[samba.SambaTensor]:
  images = samba.randn(args.mbsize, 1, args.psz, args.psz, name='image', batch_dim=0)
  labels = samba.randn(args.mbsize, 2, name='label', batch_dim=0)

  # return (images)
  return (images, labels)


def main(argv):
  args = parse_app_args(argv=argv, common_parser_fn=add_args, run_parser_fn=add_run_args)
  args.local_rank = -1

  model = BraggNN(imgsz=args.psz, fcsz=args.fcsz)
  _ = model.apply(model_init)

  # convert pytorch model to samba model
  samba.from_torch_(model)

  # optimizer = samba.optim.Adam(model.parameters(), lr=args.lr)
  optimizer = samba.optim.SGD(model.parameters(), lr=args.lr)
  
  inputs = get_inputs(args) 

  if args.command == "compile":
    samba.session.compile(model,
                          inputs,
                          optimizer,
                          name='braggNN',
                          app_dir=utils.get_file_dir(__file__))
  elif args.command == "run":
    utils.trace_graph(model, inputs, optimizer, pef=args.pef, mapping=args.mapping)
    train(args, model, optimizer)
  elif args.command == "measure-performance":
    pass
  else:
    common_app_driver(args, model, inputs, optimizer, name='braggNN', app_dir=utils.get_file_dir(__file__))


if __name__ == '__main__':
  main(sys.argv[1:])
