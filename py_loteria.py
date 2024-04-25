import pyautogui
import datetime
import sqlite3 as lite
import pandas as pd
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


def escribe_bd(df) :
    try:
        con = lite.connect('Loterias.db')
        cur = con.cursor()
        #print(df.columns.tolist())
        print(df.head(5))
        for index, row in df.iterrows():
            sql ="SELECT count(*)  from historico_loteria WHERE (loteria = '"+ row.loteria + "' AND fecha = '"+ str(row.fecha) +"')"
            print(sql)
            cur.execute(sql)
            data0 = cur.fetchone()[0]
            print ("data0:",data0)
            if data0 == 0:                
                if str(row.serie) == '' or str(row.serie) is None:
                   sserie = '000'
                else:
                   sserie = row.serie
                print("Insert ",row.fecha,"-", row.numero,"-",str(sserie),"-",row.loteria) 
                sql = "INSERT INTO  historico_loteria (fecha, numero, serie, loteria) VALUES ('"+str(row.fecha)+ "','"+ str(row.numero) + "','"+ str(sserie) + "','" + str(row.loteria) +"')"
                print(sql)                
                cur.execute(sql)
                con.commit()
        con.close()    
    except lite.Error as er:
        print('SQLite error: %s' % (' '.join(er.args)))
        print("Exception class is: ", er.__class__)
        print('SQLite traceback: ')
        exc_type, exc_value, exc_tb = sys.exc_info()
        print(traceback.format_exception(exc_type, exc_value, exc_tb))
    finally:
            if con:
                con.close()
    return

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

my_date = datetime.datetime(2011,5,3,0,0,0)
CurrentDate = datetime. datetime. now()
#CurrentDate = datetime.datetime(2011,5,16,0,0,0)
#loteria = 'cundinamarca'
#loteria = 'tolima'
#loteria = 'cruz-roja'
#loteria ='huila'
#loteria ='meta'
#loteria ='valle'
#loteria ='manizales'
#loteria ='bogota'
#loteria ='quindio'
#loteria ='risaralda'
#loteria ='medellin'
#loteria ='santander'
#loteria ='boyaca'
#loteria ='extra'
loteria ='cauca'

df = pd.DataFrame({'fecha': pd.Series(dtype='datetime64[ns]'),
                   'numero': pd.Series(dtype='str'),
                   'serie': pd.Series(dtype='str'),
                   'loteria': pd.Series(dtype='str')})


while my_date.date() < CurrentDate.date(): 
    lst = []
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
# selecciona la fecha        
    WebDriverWait(
     driver, 10).until(EC.element_to_be_clickable(
      (By.CSS_SELECTOR, "#f_cauca"))).send_keys("value", nd)
    logging.info("enter loteria fecha")
    pyautogui.press('enter')

  

# lee el numero
    numero = WebDriverWait(
     driver, 10).until(EC.element_to_be_clickable(
     (By.CSS_SELECTOR, "#u4445-4")))
    print('numero:',numero.text)
# lee la serie
    serie = WebDriverWait(
    driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#u4446-4")))   
# lee fecha
    fecha = driver.find_element(By.ID, 'f_cauca')
    fecha1 = fecha.get_attribute('value')
    print(fecha1)
    #26/07/2011
#2011-07-23
    #fecha = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#f_cauca")))
    #print('fecha:',fecha.text) 
    
#<input type="date" id="f_cauca" name="f_cauca" value="2020-09-26" class="dateresul" onchange="javascript:this.form.submit()">   

    
    lst.append(str(fecha1))
    lst.append(numero.text)
    lst.append(serie.text)
    lst.append(loteria)   
    print(lst)    
    df = df.append(pd.Series(lst, df.columns), ignore_index=True)
    #print(df)
    my_date = my_date + datetime.timedelta(days = 7)

print(df)



escribe_bd(df)