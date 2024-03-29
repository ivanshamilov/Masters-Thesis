import sys

import os
import re 
import json 

from pathlib import Path
from typing import Union, List

import polars as pl
from polars.exceptions import ColumnNotFoundError

from analysis.vars import ROOT_DIR


MAIN_DIR = f"{ROOT_DIR}/data/small_data"
BIG_DATA_DIR = f"{ROOT_DIR}/data/big_data/Keystrokes/files"


with open(f"{ROOT_DIR}/mappings/key-hand.json", "r") as f:
    KEY_HAND = json.load(f)

EXISTING_KEYS = [int(x) for x in list(KEY_HAND.keys())]

class Mapper:
    
    def __init__(self):
        self.KEY_TO_CODE = dict()
        with open(f"{ROOT_DIR}/mappings/key-codes.json", "rb") as f:
            self.KEY_TO_CODE = json.load(f)
        with open(f"{ROOT_DIR}/mappings/key-hand.json", "rb") as f:
            self.KEY_TO_HAND = json.load(f)
        self.KEY_TO_CODE["<SoS>"] = 0 # Start of Sequence
        self.KEY_TO_CODE = dict(sorted(self.KEY_TO_CODE.items(), key=lambda x: x[1]))
        self.KEY_CODES = list(self.KEY_TO_CODE.values())
        self.dict_size = len(self.KEY_TO_CODE)
        self.KEY_TO_CODE["UNKNOWN"] = -1
        self.CODE_TO_KEY = { v: k for k, v in self.KEY_TO_CODE.items() }
        self.inner_mapping = { k: i for i, k in enumerate(self.KEY_CODES) }
        self.reversed_inner_mapping = {i: k for i, k in enumerate(self.KEY_CODES)}
        
    def get_key_from_code(self, code: int):
        try:
            return self.CODE_TO_KEY[code]
        except:
            return "Code not found"
    
    def get_hand_from_code(self, code: Union[str, int]):
        if type(code) == int:
            code = str(code)
        try:
            return self.KEY_TO_HAND[code]
        except:
            return "Code not found"
    
    def get_hand_from_key(self, key: str): 
        try:
            code = self.get_code_from_key(key)
            return self.KEY_TO_HAND[str(code)]
        except:
            return "Key not found"
    
    def get_code_from_key(self, key: str):
        try:
            return self.KEY_TO_CODE[key]
        except:
            return "Key not found"
        
    def get_key_from_mapped_code(self, code: int):
        try:
            code = self.reversed_inner_mapping[code]
            return self.get_key_from_code(code)
        except:
            return "Mapped code not found"
    
    def get_mapped_code_from_key(self, key: str):
        try:
            code = self.get_code_from_key(key)
            return self.inner_mapping[code]
        except:
            return "Key not found"
    
    def get_mapped_code_from_code(self, code: int):
        try:
            return self.inner_mapping[code]
        except:
            return "Code not found"
        
    def get_code_from_mapped_code(self, code: int):
        try:
            return self.reversed_inner_mapping[code]
        except:
            return "Mapped code not found"


def calculate_features(df: pl.DataFrame, drop_timestamps: bool = True):
    try:
        new_sentences = df["TEST_SECTION_ID"] == df["TEST_SECTION_ID"].shift_and_fill(fill_value=0)
    except ColumnNotFoundError:
        new_sentences = pl.Series([False, ] + [True] * (df.shape[0] - 1))
    df = df.select([
        (pl.col("RELEASE_TIME") - pl.col("PRESS_TIME")).alias("HOLD_TIME"),
        pl.when(new_sentences).then(
            pl.col("KEYCODE").shift_and_fill(fill_value=0)
        ).otherwise(0).alias("PREV_KEYCODE"),
        pl.when(new_sentences).then(
            pl.col("PRESS_TIME") - pl.col("PRESS_TIME").shift_and_fill(fill_value=0)
        ).otherwise(0).alias("PRESS_PRESS_TIME"),
        pl.when(new_sentences).then(
            pl.col("PRESS_TIME") - pl.col("RELEASE_TIME").shift_and_fill(fill_value=0)
        ).otherwise(0).alias("RELEASE_PRESS_TIME"),
        pl.when(new_sentences).then(
            pl.col("RELEASE_TIME") - pl.col("RELEASE_TIME").shift_and_fill(fill_value=0)
        ).otherwise(0).alias("RELEASE_RELEASE_TIME"),     
        pl.when(new_sentences).then(0).otherwise(1).alias("NEW_SENTENCE"),
        pl.col("*")
    ])
    if drop_timestamps: df.drop(["RELEASE_TIME", "PRESS_TIME"])
    
    return df


def read_data_for_participant(participant_id: int, 
                              directory=MAIN_DIR,
                              preprocess: bool=True,
                              drop_timestamps: bool=True,
                              print_info: bool=False,
                              columns_to_read: List[str]=[],
                              **kwargs):
    global MAIN_DIR
    if not columns_to_read:
        df = pl.read_csv(os.path.join(directory, f"{participant_id}_keystrokes.txt"), separator="\t", infer_schema_length=10**12, **kwargs)
    else:
        df = pl.read_csv(os.path.join(directory, f"{participant_id}_keystrokes.txt"), separator="\t", columns=columns_to_read, infer_schema_length=10**12, **kwargs)
    if "LETTER" in columns_to_read:
        df = df.with_columns([pl.col("LETTER").str.to_lowercase().alias("LETTER")])
    if preprocess:
        df = calculate_features(df, drop_timestamps=drop_timestamps)
    if print_info:
        print(f"{df.filter(pl.col('SENTENCE') == pl.col('USER_INPUT'))['TEST_SECTION_ID'].n_unique()}",
              f"/ {df['TEST_SECTION_ID'].n_unique()} sentences were written correctly",
              f"by the participant {participant_id}")
        
    df = df.drop_nulls()
    return df


def create_bigrams(df: pl.DataFrame, ignore_keys: List[int] = [0, 16, 32]) -> pl.DataFrame:  # 0 - <SoS>, 8 - backspace, 16 - shift, 32 - space
    mapper = Mapper()
    df = df.select([
        pl.col("PREV_KEYCODE"),
        pl.col("KEYCODE"),
        pl.col("RELEASE_PRESS_TIME").alias("INTER_KEY_INTERVAL"),
        pl.when((~pl.col("PREV_KEYCODE").is_in(ignore_keys)) & (~pl.col("KEYCODE").is_in(ignore_keys)))
        .then(pl.struct(["PREV_KEYCODE", "KEYCODE"]).apply(
            lambda x: f"{mapper.get_key_from_code(x['PREV_KEYCODE'])};{mapper.get_key_from_code(x['KEYCODE'])}"
        ).alias("BIGRAM"))
        .otherwise(-1),
    ]).filter((pl.col("BIGRAM") != "-1") & ((pl.col("PREV_KEYCODE").is_in(EXISTING_KEYS)) & (pl.col("KEYCODE").is_in(EXISTING_KEYS))))
    return df


def find_all_participants(directory: str) -> List[str]:
  return [re.findall(r"^[0-9]+", f)[0] for f in os.listdir(directory) if re.findall(r"^[0-9]+", f)]
