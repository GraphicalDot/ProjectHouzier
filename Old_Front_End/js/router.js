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
		var str = new App.RootView()
		//var str = new App.WordCloudWith_D3({model: {"el": this.el}})
	},
});
App.boot = function(container){
	container = $(container);
	var router = new App.Router({el: container});
	Backbone.history.start();
}
});
