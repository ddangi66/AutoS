'''# This will not run on online IDE
import requests
from bs4 import BeautifulSoup

URL = "http://www.values.com/inspirational-quotes"
r = requests.get(URL)

soup = BeautifulSoup(r.content,'html parser')  # If this line causes an error, run 'pip install html5lib' or install html5lib
print(soup.prettify())

import flask
from flask import Flask,render_template,request
import requests
#app=Flask(__name__)
from bs4 import BeautifulSoup
def rndr():
    url="https://www.google.com";
    r=requests.get(url)
    html_page=r.content
    soup=BeautifulSoup(html_page,'html.parser')
    ll=soup.text;
    ll=str(ll)
    print(ll)
#@app.route('/summary', methods=['POST' , 'GET'])

   # return render_template('summary.html', output=output,title=title)
rndr()
'''
'''
String="asdasjdkads'www.google.com'"
String = text
ll = String.find("'")
est = String[ll + 1:]
est = est[:len(est) - 1]
'''
String="ahsjhasjdnasjd'www.google.com'"
ll = String.find("'")



def nws(String,ll):
    from bs4 import BeautifulSoup
    import requests
    est = String[ll + 1:]
    str = est[:len(est) - 1]

    sq = 'https://' + str;
    url = 'https://www.google.com';
    if (sq == url):
        r = requests.get(url)
        html_page = r.content
        soup = BeautifulSoup(html_page, 'html.parser')
        ll = soup.text;
        print(ll)
    else:
        print("not able")
#ss=est[:est.len-1]
#print(ss)

nws(String,ll);

'''#text=request.form['document1']
            String1=text
            ll = String1.find("'")
            def strr(String1,ll):
                est = String1[ll + 1:]
                st = est[:len(est) - 1]
                sq = str('https://' + st)
                url1='https://www.nytimes.com';
                url2='https://www.edition.cnn.com';
                url3='https://www.news18.com'
                if(sq==url1):
                    return '1';
                elif(sq==url2):
                    return '2';
                elif(sq==url3):
                    return '3'''
q = strr(String1, ll);
url1 = 'https://www.nytimes.com';
url2 = 'https://www.edition.cnn.com';
url3 = 'https://www.news18.com'


def nws(String1, ll, url):
    # url = 'https://www.indiatoday.in';

    est = String1[ll + 1:]
    st = est[:len(est) - 1]
    sq = str('https://' + st)
    # url = 'https://www.google.com';
    r = requests.get(url)
    html_page = r.content
    soup = BeautifulSoup(html_page, 'html.parser')
    ll11 = soup.text;
    return ll11


# pl=strr(url,String1,ll);
if (q is '1'):
    text = nws(String1, ll, url1)
elif (q is '2'):
    text = nws(String1, ll, url2)
elif (q is '3'):
    text = nws(String1, ll, url3)
