# -*- coding: utf-8 -*-
"""P25 公司行动交叉核验命令入口；实际规则与全量回灌共用同一实现。"""
import json

import pandas as pd

from execution_replay import SCORES, RawStore, validate_corporate_actions


def main():
    instruments = set()
    for fold in range(8, 12):
        s = pd.read_parquet(SCORES / f"alstm_fold{fold}.parquet", columns=["instrument"])
        instruments.update(s["instrument"].str.upper())
    summary = validate_corporate_actions(RawStore(instruments))
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
