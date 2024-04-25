import datetime

my_date = datetime.datetime(2011,5,2,0,0,0)


CurrentDate = datetime. datetime. now()


while my_date.date() < CurrentDate.date():
    my_date = my_date + datetime.timedelta(days = 7)
    print(my_date) 