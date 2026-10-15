# Return-schema infrastructure repair

首次数值fixture在forward unpack发现ValueError too many values：raw-controls保存扩展错改forward的三返回值而未改generated的四返回值。实际FIT数据/sourceID/GPU8/2epoch/benchmark尚未运行，无生产worker或科学结果。全部首次七源码保留failures/preflight1_return_schema，root唯一PREFLIGHT_FAILURE来源保留不覆盖。

仅恢复forward(out,res,u)与generated(out,res,u,raw)的既定接口。两arm目标、结构、输出数值、阈值、FD eps和训练协议不变。增加已失败来源到继承source锁，未替换任何旧digest。完整重跑预检，不将首次失败混作生产训练失败，不声称预检已PASS。
