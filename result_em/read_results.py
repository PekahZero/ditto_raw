'''
import os
from tensorboard.backend.event_processing import event_accumulator
# from tensorflow.python.summary.summary_iterator import summary_iterator
from glob import glob

rank_field = 'f1'
fields = ['t_acc', 't_f1']
runs = glob("*/") # 匹配当前目录下的所有子目录
results = {}

def get_latest(events):
  best = 0.0
  res = events[0]
  for event_fn in events:
    tm = os.path.getmtime(event_fn)
    if tm > best:
      best = tm
      res = event_fn
  return res


for run in runs:
  try:
    target_events = glob('%s%s/*' % (run, rank_field))
    # glob() 函数接受这个格式化的字符串作为参数,并返回与该模式匹配的所有文件路径的列表
    values = []

    event_fn = get_latest(target_events)
    for e in summary_iterator(event_fn):
      for v in e.summary.value:
        values.append(v.simple_value)

    for field in fields:
      target_events = glob('%s%s/*' % (run, field))
      o_values = []
      event_fn = get_latest(target_events)
      for e in summary_iterator(event_fn):
        for v in e.summary.value:
          o_values.append(v.simple_value)
      max_val = 0.0
      result = 0.0
      for v, ov in zip(values[5:], o_values[5:]):
        if v > max_val:
          max_val = v
          result = ov

      print(run, field, result)

      if run not in results:
        results[run] = {}
      results[run][field] = result
  except:
    pass
  
'''

import os

# from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.backend.event_processing import event_accumulator
from glob import glob

rank_field = 'f1'  # 用于排序的字段
fields = ['t_acc', 't_f1']  # 需要提取的字段
runs = glob("*/")  # 获取所有的运行目录
results = {}  # 存储结果的字典

def get_latest(events):
  """
  获取最新的事件文件
  :param events: 事件文件列表
  :return: 最新的事件文件
  """
  best = 0.0
  res = events[0]
  for event_fn in events:
    tm = os.path.getmtime(event_fn)  # 获取文件的修改时间
    if tm > best:
      best = tm
      res = event_fn
  return res


for run in runs:  # 遍历每个运行目录
  try:
    target_events = glob('%s%s/*' % (run, rank_field))  # 获取排序字段的事件文件
    values = []  # 存储排序字段的值

    # event_fn = get_latest(target_events)  # 获取最新的事件文件
    # for e in summary_iterator(event_fn):  # 遍历事件文件
    #   for v in e.summary.value:
    #     values.append(v.simple_value)  # 提取排序字段的值

#----------------tenssorboard-------
    event_fn = get_latest(target_events)
    event_acc = event_accumulator.EventAccumulator(event_fn)
    event_acc.Reload()
    scalars = event_acc.Scalars(rank_field)
    values = [s.value for s in scalars]

    for field in fields:  # 遍历需要提取的字段
      target_events = glob('%s%s/*' % (run, field))  # 获取字段的事件文件
      
      # o_values = []  # 存储字段的值
      # event_fn = get_latest(target_events)  # 获取最新的事件文件
      # for e in summary_iterator(event_fn):  # 遍历事件文件
      #   for v in e.summary.value:
      #     o_values.append(v.simple_value)  # 提取字段的值
    
      event_fn = get_latest(target_events)
      event_acc = event_accumulator.EventAccumulator(event_fn)
      event_acc.Reload()
      scalars = event_acc.Scalars(rank_field)
      o_values = [s.value for s in scalars]    
          
      max_val = 0.0
      result = 0.0
      for v, ov in zip(values[5:], o_values[5:]):  # 遍历排序字段和字段的值
        if v > max_val:  # 如果排序字段的值更大
          max_val = v
          result = ov  # 更新结果为对应的字段值

      print(run, field, result)  # 打印运行目录、字段和结果

      if run not in results:
        results[run] = {}  # 如果运行目录不在结果字典中,则添加一个空字典
      results[run][field] = result  # 将结果存储到结果字典中
  except:
    pass  # 如果出现异常,则忽略
