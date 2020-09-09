# -*- encoding: utf8 -*-
import fire
import logging
from capture import *
from train import train_model

def catch(tag):
    catch_video(tag)
    logging.info("catch_face done.")

def train():
    train_model()
    logging.info("train done.")

if __name__ == "__main__":
    print(config.PROJECT_PATH)
    catch(1)
    # train()
    # logging.getLogger().setLevel(logging.INFO)
    # fire.Fire()