#coding:utf-8
import urllib
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8') #输出的内容是utf-8格式

#打开网址,获取源码
def get_content(page):
    url = url0.format(page)
    a = urllib.urlopen(url) #打开网址
    html = a.read() #读取源代码
    html = html.decode('gbk') #从gbk转为unicode
    #print html
    return html

#匹配到正文
def get(html):
    reg = re.compile(r'class="t1 ">.*?<a target="_blank" title="(.*?)".*?<span class="t2"><a target="_blank" title="(.*?)".*?<span class="t3">(.*?)</span>.*?<span class="t4">(.*?)</span>.*?<span class="t5">(.*?)</span>',re.S)
    items = re.findall(reg,html)
    #print items #列表list
    return items

#多页,写入文件
for j in range(1,2000):

    #url0 = 'http://search.51job.com/jobsearch/search_result.php?fromJs=1&jobarea=000000%2C00&district=000000&funtype=0000&industrytype=00&issuedate=9&providesalary=99&keywordtype=2&curr_page='+str(j)+'&lang=c&stype=1&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&lonlat=0%2C0&radius=-1&ord_field=0&list_type=0&fromType=14&dibiaoid=0&confirmdate=9'
    #url0 = 'http://search.51job.com/list/000000,000000,2501%252C2502%252C2514%252C2537%252C2539,01%252C32,9,99,%2B,2,' + str(j) + '.html?lang=c&stype=1&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&lonlat=0%2C0&radius=-1&ord_field=0&confirmdate=9&fromType=1&dibiaoid=0&address=&line=&specialarea=00&from=&welfare='
    url0 = 'http://search.51job.com/list/000000,000000,0000,00,9,99,%2B,2,' + str(j) + '.html?lang=c&stype=1&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&lonlat=0%2C0&radius=-1&ord_field=0&confirmdate=9&fromType=1&dibiaoid=0&address=&line=&specialarea=00&from=&welfare='
    html = get_content(url0)  # 调用获取源码
    for i in get(html):

        with open('data-all.txt','a') as f:
            f.write(i[0]+'\t'+i[1]+'\t'+i[2]+'\t'+i[3]+'\n')

