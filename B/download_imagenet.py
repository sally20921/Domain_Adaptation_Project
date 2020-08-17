import  urllib.request as url
import os
import numpy as np
import requests
import urllib3
from PIL import Image
from io import  BytesIO
from bs4 import BeautifulSoup
import timeit
from multiprocessing import Pool
from pathlib  import  Path

DIRECTORY = './image_url'
DIRECTORY_IMAGE = './datasets'

#link = os.listdir('./image_url')[1]

def write_content(directory, content_name, wid):
  if not os.path.isdir(directory):
    os.mkdir(directory)

  file_loc = directory+'/'+content_name
  my_file = Path(file_loc)
  base_url = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid='
  
  if not my_file.is_file():
    try: 
      web_page = requests.get(base_url+wid, timeout=44)
      with open(file_loc, 'w') as file:
        file.write(web_page.text)
    except requests.RequestException as e:
      print(e)

def grab_inter(wid):
  write_content(DIRECTORY, wid+'.txt', wid)

def grabber(link):
  with open(DIRECTORY+'/'+link, 'r') as file_link:
    i = 0
    directory_image_temp = DIRECTORY_IMAGE+'/'+link.replace('.txt', '')
    print(directory_image_temp)

    if not os.path.isdir(directory_image_temp):
      os.mkdir(directory_image_temp)

    for line in file_link:
      try:
        webpage  = requests.get(line, timeout=44)
        image = Image.open(BytesIO(webpage.content))
        image.save(directory_image_temp+'/'+str(i)+'.jpg')
        i = i+1
      except OSError:
        print("OSError")
        pass

def main():
  list_array = []

  with open('./imagenet_synset_list.txt', 'r') as file:
    for line  in file:
      list_array.append(line.replace('\n', ''))

  print("Downloading  Imagenet Dataset!")

  #with Pool(50) as p:
  #  p.map(grab_inter, list_array)
  for wib in list_array:
    grab_inter(wib)
  print("Finished downloading Dataset!")
  
  links = os.listdir(DIRECTORY)
  #with Pool(processes=50) as p:
  #  p.map(grabber, links)
  for link in links:
    grabber(link)

if __name__ == '__main__':
  main()
  
