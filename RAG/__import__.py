import json
import mmap
import multiprocessing
import os
import smtplib
import socket
import time
import unicodedata
import warnings
from collections import defaultdict
from email.message import EmailMessage
from functools import lru_cache
from typing import Literal

# from langchain_community.llms import OpenAI
import google.generativeai as genai
import ijson
import pandas as pd
import regex as re
import requests
import spacy
import torch
import urllib3
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer
