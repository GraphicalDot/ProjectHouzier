$(document).ready(function(){
App.Router = Backbone.Router.extend({
	initialize: function(options){
		this.el =  options.el ;
		console.log(this.el)
	},

	routes: {
		"":  "welcome",
	},
	
	welcome: function(){
		var str = new App.RootView({model: {"el": this.el}})
		//var str = new App.WordCloudWith_D3({model: {"el": this.el}})
		this.el.html(str.render().el);
	},
});
App.boot = function(container){
	container = $(container);
	var router = new App.Router({el: container});
	Backbone.history.start();
}
});
