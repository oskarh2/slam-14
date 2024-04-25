import urllib.request
from pprint import pprint
from html_table_parser.parser import HTMLTableParser
import sqlite3 as lite
import pandas as pd

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

def fmonth(i):
        switcher={
                'enero':'01',
                'febrero':'02',
                'marzo':'03',
                'abril':'04',
                'mayo':'05',
                'junio':'06',
                'julio':'07',
                'agosto':'08',
                'septiembre':'09',
                'octubre':'10',
                'noviembre':'11',
                'diciembre':'12'
             }
        return switcher.get(i,"Invalid month of year")


def conv_fecha(st):
    st = st.split(' ')
    dia = st[1]
    ano = st[3]
    mes = fmonth(st[2])
    fecha = dia +'/'+mes+'/'+ ano
    return fecha

def url_get_contents(url):
    req = urllib.request.Request(url=url)
    f = urllib.request.urlopen(req)
    return f.read()

def loteria(st):
    slink = 'https://www.astroluna.co/' + st 
    xhtml = url_get_contents(slink).decode('utf-8')
    p = HTMLTableParser()
    p.feed(xhtml)
    x = p.tables[0]
    df = pd.DataFrame(x[1:],columns=x[0])
    df['fecha'] = df['Fecha'].apply(conv_fecha)
    df['fecha'] = pd.to_datetime(df['fecha'].str.strip(), format='%d/%m/%Y')
    df.rename(columns = {'Número':'numero', 'Serie':'serie'}, inplace = True)
    df = df.assign(loteria =lambda x : st)
    return df

dfloterias = pd.DataFrame({'fecha': pd.Series(dtype='datetime64[ns]'),
                   'numero': pd.Series(dtype='str'),
                   'serie': pd.Series(dtype='str'),
                   'loteria': pd.Series(dtype='str')})
    

#sloterias = ['cundinamarca', 'cruz-roja', 'valle', 'bogota', 'medellin','boyaca', 'tolima','huila', 'meta', 'manizales', 'quindio','santander','risaralda','cauca']
sloterias = ['boyaca']



for i in sloterias:
    print(i)
    dfloteria = loteria(i)
    dfloterias = dfloterias.append(dfloteria, ignore_index=True)

print(dfloterias)
    

#dfloterias.to_excel(r'loterias.xlsx', index = False, header = True)


escribe_bd(dfloterias)
