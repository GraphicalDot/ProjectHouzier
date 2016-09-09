#!/usr/bin/env ipython
#-*- coding: utf-8 -*-
import sys
import os
import requests
import lxml.html
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import Select
from redis_storage import RedisProxy

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent)

class XRoxyProxies:
	"""
	This is the class used to extract and populate proxies from different website
	Proxies
	Usage:
		instance = Proxies(number_of_proxies=100, type_of_proxy="Socks4", country="CN", latency="3000", reliability="9000")
		then call instance.xroxy_in_browser() to get the proxy_list 

	Args:
		number_of_proxies:
			type int, number of proxies to be populated
		type_of_proxy:
			type str, type of proxy to be populated socks4 or socks5
		country:
			type str, proxies if required from a particular country india, us, uk etc
			(u'Any country', u''), (u'Afghanistan', u'AF'), (u'Albania', u'AL'), (u'Argentina', u'AR'), 
			(u'Armenia', u'AM'), (u'Bangladesh', u'BD'), (u'Belarus', u'BY'), (u'Belgium', u'BE'), (u'Benin', u'BJ'), 
			(u'Bolivia', u'BO'), (u'Bosnia and Herzegovina', u'BA'), (u'Brazil', u'BR'), (u'Bulgaria', u'BG'), 
			(u'Cambodia', u'KH'), (u'Cameroon', u'CM'), (u'Canada', u'CA'), (u'Chad', u'TD'), (u'Chile', u'CL'), 
			(u'China', u'CN'), (u'Colombia', u'CO'), (u"Cote D'Ivoire (Ivory Coast)", u'CD'), (u'Croatia (Hrvatska)', u'HR'), 
			(u'Cuba', u'CU'), (u'Cyprus', u'CY'), (u'Czech Republic', u'CZ'), (u'Denmark', u'DK'), (u'Ecuador', u'EC'), 
			(u'Egypt', u'EG'), (u'El Salvador', u'SV'), (u'Ethiopia', u'ET'), (u'European Union', u'EU'), (u'France', u'FR'), 
			(u'Georgia', u'GE'), (u'Germany', u'DE'), (u'Ghana', u'GH'), (u'Great Britain (UK)', u'GB'), (u'Greece', u'GR'), 
			(u'Honduras', u'HN'), (u'Hong Kong', u'HK'), (u'Hungary', u'HU'), (u'India', u'IN'), (u'Indonesia', u'ID'), 
			(u'Iran', u'IR'), (u'Iraq', u'IQ'), (u'Israel', u'IL'), (u'Italy', u'IT'), (u'Japan', u'JP'), (u'Kazakhstan', u'KZ'), 
			(u'Kenya', u'KE'), (u'Korea (South)', u'KR'), (u'Kuwait', u'KW'), (u'Lebanon', u'LB'), (u'Liberia', u'LR'), 
			(u'Lithuania', u'LT'), (u'Luxembourg', u'LU'), (u'Madagascar', u'MG'), (u'Malaysia', u'MY'), (u'Mali', u'ML'), 
			(u'Mauritius', u'MU'), (u'Mexico', u'MX'), (u'Mongolia', u'MN'), (u'Mozambique', u'MZ'), (u'Myanmar', u'MM'), 
			(u'Nepal', u'NP'), (u'Netherlands', u'NL'), (u'Nigeria', u'NG'), (u'Pakistan', u'PK'), (u'Palestine', u'PS'), 
			(u'Panama', u'PA'), (u'Paraguay', u'PY'), (u'Peru', u'PE'), (u'Philippines', u'PH'), (u'Poland', u'PL'), 
			(u'Puerto Rico', u'PR'), (u'Romania', u'RO'), (u'Russian Federation', u'RU'), (u'Saudi Arabia', u'SA'), 
			(u'Singapore', u'SG'), (u'Slovak Republic', u'SK'), (u'Slovenia', u'SI'), (u'South Africa', u'ZA'), (u'Spain', u'ES'), 
			(u'Sri Lanka', u'LK'), (u'Sudan', u'SD'), (u'Sweden', u'SE'), (u'Switzerland', u'CH'), (u'Taiwan', u'TW'), 
			(u'Tanzania', u'TZ'), (u'Thailand', u'TH'), (u'Tunisia', u'TN'), (u'Turkey', u'TR'), (u'Ukraine', u'UA'), 
			(u'United Arab Emirates', u'AE'), (u'United States', u'US'), (u'Vanuatu', u'VU'), (u'Venezuela', u'VE'), 
			(u'Viet Nam', u'VN'), (u'Zambia', u'ZM'), (u'Zimbabwe', u'ZW')

		latency:
			type str, latency required in a proxy, values can be 1000, 3000, 5000, or 10000, 
			which corresponds to less than 1 sec, leass than 5 sec, less than 3 sec and less
			than 10 sec respectively.

		reliability:
			type str, reliability of the proxy, values can be 2500, 5000, 7500, 9000, 
			which correponds to more than 25%, more than 50%, more than 75% and more than 90 %
		


	Returns:
		list of proxies whose number will be equal to the number_of_proxies argument, depending upon the 
		availability of the proxies in the defined paramteres in the arguments

	"""
	def __init__(self, number_of_proxies, type_of_proxy=None, country=None, latency=None, reliability=None):
		self.arguments = lambda x: x if x else None
		self.number_of_proxies = self.arguments(number_of_proxies)
		self.type_of_proxy = self.arguments(type_of_proxy)
		self.country = self.arguments(country)
		self.latency = self.arguments(latency)
		self.reliability = self.arguments(reliability)
		self.proxy_list = list()
		self.xroxy_in_browser()
	
		self.redis()

	def __xroxy_parse(self, html):
		"""
		This method gets the page of the xroxy website with params like latency, country etc appended to it.
		It then parses the page to get two results
		1.Proxy_list
		2.Page_list

		If the length of the proxy list is less than the number_of_proxies then it parses the next page fron the
		page list to get more proxies, It repeats this process untill the length of proxy_list increases or becomes
		equal to the number_of_proxies param of this class.

		Returns:
			proxy_list: Whose length is equal to or greater than number_of_proxies param of this class

		"/proxylist.php?port=&;type={proxy_type}&ssl=&country={country}&latency={latency}&reliability={reliability}&
		sort=reliability&desc=true&amp&pnum={page_number}#table".format(proxy_type=self.type_of_proxy, page_number=page 
		country=self.country, latency=self.latency, reliability=self.reliability) for page in 
		"""
		#this when provided with html of a page returns a lxml tree
		html_tree = lambda html: lxml.html.fromstring(html)

		#this when provided with lxml tree returns the list of proxies present in the lxml tree for xroxy website
		#proxies = lambda tree:  [proxy.split()[0] for proxy in tree.xpath("//tr[starts-with(@class, 'row')]/td[2]/a/text()[1]")]
		proxies = lambda tree:  [{"ip": proxy.xpath("td[2]/a/text()")[0].split()[0], 
					"port": proxy.xpath("td[3]/a/text()")[0],
					"type": proxy.xpath("td[4]/a/text()")[0],
					"country": proxy.xpath("td[6]/a/small/text()")[0],
					"latency": proxy.xpath("td[7]/text()")[0],
					"reliability": proxy.xpath("td[8]/text()")[0],}
				for proxy in tree.xpath("//tr[starts-with(@class, 'row')]")]
		
		
		def scrape(url):
			r = requests.get(url)
			if r.status_code == 200:
				tree = html_tree(r.text)
				self.proxy_list.extend(proxies(tree))
				return True
			return False

		tree = html_tree(html) #Making lxml tree from the html source provided to this method as an argument
		
		self.proxy_list.extend(proxies(tree)) #adding proxies from the first page to the self.proxy_list

		
		


		pages = self.pages(tree, html_tree)	

		page_number = 0
		while len(self.proxy_list) <= self.number_of_proxies: 
			try:
				if scrape(pages[page_number]):
					page_number += 1			
				else:
					raise StandardError("The page %s cannot be fetched"%pages_list[page_number])
			except IndexError as e:
				raise StandardError(e)
				return 
		return 

	def pages(self, tree, html_tree):
		"""
		This method populaf the pagte the pages list for the query.
		How it works:
			The problem is if the total number of queries is around 126, The xroxy will display only 4 pages as the pagination.
			The rest of the pages will only be shown, when you go to page 4, So from the html of the main page, we will get
			the 4 pages, then get the 4th page to get the next set of pages, till the pagination ends.

			Pages upto 4 will be appended to a list, then the html for the last element of the list will be fetched, appended to the 
			list, till pagination ends.
			This will also have duplicacy of the links, so the list will be converted to the set, then converted back to a list
			to eliminate duplicacy of the links
		"""
		pages_list = ["http://www.xroxy.com/%s"%element for element in 
					tree.xpath("//table[@class='tbl']/tbody/tr/td[1]/table/tbody/tr/td/small/a/@href")[1:]]
		
		is_pages = True
		last_link = str()

		while is_pages:
			if last_link == pages_list[-1]:
				is_pages = False
			
			try:
				driver = webdriver.Firefox()
				driver.get(pages_list[-1])
				last_link = pages_list[-1]
			
			except Exception as a:
				raise StandardError(e)
			
			html = driver.page_source
			driver.close()
			child_tree = html_tree(html)
			pages_list.extend(["http://www.xroxy.com/%s"%element for element in 
					child_tree.xpath("//table[@class='tbl']/tbody/tr/td[1]/table/tbody/tr/td/small/a/@href")])
			
		return list(set(pages_list))


	def xroxy_in_browser(self):
		"""
		This class method opens a browser with xroxy website and selects the option provied in the args of this 
		class  to filter the proxies
		
		Returns:
			A proxy_list 
		"""

		print "this is the number of proxies recuired %s"%self.number_of_proxies
		driver = webdriver.Firefox()
		driver.get("http://www.xroxy.com/proxylist.htm")
		
		if self.type_of_proxy:
			self.__select_option(driver, "//select[@id='type_id']", self.type_of_proxy)
		
		if self.country:
			self.__select_option(driver, "//select[@id='country_id']", self.country)
		
		if self.latency:
			self.__select_option(driver, "//select[@id='latency_id']", self.latency)
		
		if self.reliability:
			self.__select_option(driver, "//select[@id='reliability_id']", self.reliability)
		
		html = driver.page_source
		driver.close()
		self.__xroxy_parse(html) #here we have the whole proxy list
		#self.redis(proxy_list)
		return True


	def __select_option(self, driver, xpath, option):
		"""
		This method clicks on the webpage present on the selenium webdriver on the basis of the xpath of 
		the select option present on the dom of the webpage

		"""
		element = driver.find_element_by_xpath(xpath)
		p = Select(element)
		try:
			p.select_by_value(option)
		except TypeError as e: #None type cannot be passed as a value into the select_by_value option so it doesnt change anything on browser
			pass
		return
	

	def redis(self):
		"""
		This method will take a proxy_list and then store it in the redis.
		In case, Unable to store in redis, A standard error will be raised with the exception messege from the class RDS
		which will then be handled here
		
		All the proxies scraped from xroxy will be put into redis with staus "unhealthy" as we would be not clear
		whether they are healthy or not, We will have to check them with our ProxyHealthCheck.
		
		"""
		redis_instance = RedisProxy()
		try:
			redis_instance.store_proxy_list(self.proxy_list, "unhealthy")
		except StandardError as e:
			print e



if __name__ == "__main__":
	#instance = XRoxyProxies(number_of_proxies=100, type_of_proxy="Socks4", country="CN")
	instance = XRoxyProxies(number_of_proxies=100, type_of_proxy="Socks4")
	print instance.proxy_list
	
	#instance = XRoxyProxies(number_of_proxies=10)
