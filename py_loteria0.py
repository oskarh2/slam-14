import pyautogui
import datetime
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
#from selenium.webdriver.support.expected_conditions import presence_of_element_located
from selenium.webdriver.support import expected_conditions as EC
import logging
import time
import sys
import re
#url =  "https://news.google.com/topstories?hl=es-419&gl=CO&ceid=CO:es-419"

url =  "https://www.pagatodo.com.co/resultados.php?plg=resultados-loterias#ac1"
chrome_driver_path = 'c:/python37/chromedriver.exe'


# Gets or creates a logger
logger = logging.getLogger(__name__)  
logging.basicConfig(level=logging.INFO, filename='loteria.log',format='%(asctime)s %(levelname)s:%(message)s')
logging.info("Programa para buscar extraer informacion usando Python y librerias como Selenium, csv, re, bs4")
logging.info("Inicia Cargando webdriver.Chrome")
driver = webdriver.Chrome()
chrome_path = 'C:/python37/chromedriver.exe'
driver = webdriver.Chrome(chrome_path)
# open the webpage
logging.info("open the webpage")

driver.get(url)
# target loteria
#cundinamarca, tolima
#my_date = datetime.datetime(2011,5,2,0,0,0)
#cruzroja

my_date = datetime.datetime(2011,5,3,0,0,0)

CurrentDate = datetime. datetime. now()


print(my_date)

if len(str(my_date.day)) == 1:
   nday = '0'+str(my_date.day)
else:
   nday = str(my_date.day)
if len(str(my_date.month)) == 1:
   nmonth = '0'+str(my_date.month)
else:
   nmonth = str(my_date.month)   
nd = nday +'/'+nmonth+'/'+str(my_date.year)
print(nd)

#f_cundinamarca
#f_tolima
#f_cruzroja
#f_huila

#f_meta
#f_valle
#f_manizales
#f_bogota

#f_quindio
#f_risaralda
#f_medellin
#f_santander

#f_boyaca
#f_extra
#f_cauca

WebDriverWait(
	driver, 10).until(EC.element_to_be_clickable(
		(By.CSS_SELECTOR, "#f_huila"))).send_keys("value", nd)
logging.info("enter loteria fecha")
pyautogui.press('enter')
#u3292-4
#u3215-4
#u929-4
#u3130-4

#u3714-4
#u3781-4
#u3859-4
#u3944-4

#u4026-4
#u4114-4
#u4199-4
#u4372-4

#u4278-4
#u4534-4
#u4445-4


valor = WebDriverWait(
	driver, 10).until(EC.element_to_be_clickable(
		(By.CSS_SELECTOR, "#u3130-4")))

#u3305-4
#u3219-4
#u931-4
#u3143-4

#u3717-4
#u3790-4
#u3872-4
#u3948-4

#u4033-4
#u4111-4
#u4201-4
#u4361-4

#u4269-4
#u4516-4
#u4446-4
serie = WebDriverWait(
	driver, 10).until(EC.element_to_be_clickable(
		(By.CSS_SELECTOR, "#u3143-4")))
print('valor:',valor.text,' serie:',serie.text)
    
    
    
    
