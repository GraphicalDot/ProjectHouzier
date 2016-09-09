#-*- coding: utf-8 -*-
import redis

class RedisProxy:

	def __init__ (self, db=None):
		"""
		This class deals with the storing, creating, deleting, getting object from the proxy_list stored in the
		redis database.
		Args:
			
			For unhealthy_proxies pass redis_list_name = "unhealthy_proxies" as an argument when initiating this class

		"""
		if not db:
			self.redis_connection = redis.StrictRedis(host='localhost', port=6379, db=15)
		else:
			self.redis_connection = redis.StrictRedis(host='localhost', port=6379, db=db)

		
	def store_proxy_list(self, proxy_list, status):
		"""
		proxy_list is the list of the proxies which will be stored in the redis proxies list
		Each element is in the form of 
		{"ip": ip, "port": 1080, "type": Socks4, "country": Brazil, "latency": 30, "reliability": 90}

		status: healthy or unhealhty
		if status != "healthy":
			raise StandardError("not a valid status for proxy")
		
		if status != "unhealthy":
			raise StandardError("not a valid status for proxy")
		"""
		with self.redis_connection.pipeline() as pipe:
			try:
				for proxy in proxy_list:
					proxy["status"] = status
					pipe.hmset(proxy.get("ip"), proxy)
				pipe.execute()
			except Exception as e:
				raise StandardError(e)

	def total_proxies(self):
		proxy_list = self.redis_connection.keys()
		return proxy_list


	def proxy_details(self, proxy):
		"""
		Return keys and its values for the related proxy
		"""
		proxy_details = self.redis_connection.hgetall(proxy)
		return proxy_details


	def delete_proxy(self, proxy):
		"""
		Delete proxy
		"""
		self.redis_connection.delete(proxy)
		return 


	def healthy_proxies(self):
		"""
		returns the list of healthy proxies of the form
		[{'country': '\xc2\xa0Mexico', 'ip': '187.163.164.233', 'latency': '30', 'port': '1080', 
		'reliability': '100', 'status': 'healthy', 'type': 'Socks4'},

		{'country': '\xc2\xa0Pakistan', 'ip': '221.120.222.69', 'latency': '30', 'port': '1080', 
		'reliability': '100', 'status': 'healthy', 'type': 'Socks5'}]

		"""
		proxy_list = [self.redis_connection.hgetall(key) for key in self.redis_connection.keys() 
				if self.redis_connection.hget(key, "status") == "healthy"]


		return proxy_list
	
	def unhealthy_proxies(self):
		"""
		returns the list of unhealthy proxies
		[{'country': '\xc2\xa0Mexico', 'ip': '187.163.164.233', 'latency': '30', 'port': '1080', 
		'reliability': '100', 'status': 'unhealthy', 'type': 'Socks4'},

		{'country': '\xc2\xa0Pakistan', 'ip': '221.120.222.69', 'latency': '30', 'port': '1080', 
		'reliability': '100', 'status': 'unhealthy', 'type': 'Socks5'}]
		"""
		proxy_list = [self.redis_connection.hgetall(key) for key in self.redis_connection.keys() 
				if self.redis_connection.hget(key, "status") == "unhealthy"]
		return proxy_list


	def update_status(self, ip, status):
		"""
		This method updates the status of the proxy present in the database
		"""
		if not status in ("healthy", "unhealthy"):
			raise StandardError("Status that has been provided is not a valid one")

		try:
			self.redis_connection.hset(ip, "status", status)

		except Exception as e:
			raise StandardError(e)



