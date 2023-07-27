from WBF import make_WBF_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--source', type=str, help='img dir path')
parser.add_argument('-d', '--dataset', type=str, help='dataset name')
parser.add_argument('-p','--pass-conf', type=float, help='pass confidence')
parser.add_argument('-a','--amb-conf', type=float, help='amb confidence')
parser.add_argument('-f','--fail-conf', type=float, help='fail confidence')
opt = parser.parse_args()

make_WBF_file(opt.source, opt.dataset,opt.pass_conf, opt.amb_conf, opt.fail_conf)