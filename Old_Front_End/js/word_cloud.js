$(document).ready(function(){



App.SeeWordCloudDateSelectionView = Backbone.View.extend({
	template: window.template("see-word-cloud-date-selection"),
	tag: "form",
	className: "form-horizontal",
	initialize: function(options){
		function newfunction(){
			function another(){
				console.log(this)
			}
			another()
		}
		console.log(this)
		newfunction()
	},


	render: function(){
		this.beforeRender();
		this.$el.append(this.template(this));
		return this;	
	},

	beforeRender: function(){
		var self = this;
		window.eatery_id = $(":checkbox:checked").attr("id");
	
		/*	
		var jqhr = $.post(window.get_start_date_for_restaurant, {"eatery_id": window.eatery_id})	
		jqhr.done(function(data){
			console.log(data.result)
			self.$("#startDate").val(data.result.start)
			self.$("#selectStartDate").val(data.result.start)
			self.$("#endDate").val(data.result.end)
			self.$("#selectEndDate").val(data.result.end)
		});
		*/
		},


	events: {
		"click #idSubmit": "submit",
		"click #idCancel": "cancel",

	},

	loading_bootbox: function(){
		$(".data_selection").modal("hide");
		bootbox.dialog({ 
			closeButton: false, 
			message: "<img src='css/images/y4RRIFJ.jpg'>",
			className: "loadingclass",
		});
	},

	submit: function(event){
		var self = this;
		$(".dynamic_display_word_cloud").empty();	
		event.preventDefault();
		this.$el.addClass("dvLoading");
	
		window.word_cloud_category =  $("#wordCloudCategory").find("option:selected").val();

		console.log("This is the category selected" + window.word_cloud_category);	
		/*
		if (window.word_cloud_category == "Service"){
			bootbox.alert("The service word cloud has not been implemented yet !!");
			$(".loadingclass").modal("hide");
			$(".data_selection").modal("hide");
			$.each($(":checkbox"), function(iter, __checkbox){
				$("#" + __checkbox.id).prop("checked", false); 
				})
			return 
		}

		*/
		this.loading_bootbox()
		/*
		if ($('#startDate').val() > $('#selectStartDate').val()){
			bootbox.alert("Genius the start date selected should be greater then start date").find('.modal-content').addClass("bootbox-modal-custom-class");
			return

		}
		if ($('#endDate').val() < $('#selectEndDate').val()){

			bootbox.alert("Genius the end date selected should be less then end date").find('.modal-content').addClass("bootbox-modal-custom-class");
			return
		}
	
		*/
		//payload = {"eatery_id": eatery_id, "category":"overall", "total_noun_phrases": 15, "word_tokenization_algorithm": 
		//	'punkt_n_treebank', "pos_tagging_algorithm": "hunpos_pos_tagger", "noun_phrases_algorithm":"regex_textblob_conll_np"}
		


		console.log(window.eatery_id + "    " + window.word_cloud_category);	
		var jqhr = $.post(window.get_word_cloud, {"eatery_id": window.eatery_id,
		//					"start_date": $('#selectStartDate').val(),
		//					"end_date": $('#selectEndDate').val(),
		    					"category": window.word_cloud_category,
							"total_noun_phrases": 20,	
					})


		
		//On success of the jquery post request
		jqhr.done(function(data){
			var subView = new App.WordCloudWith_D3({model: data.result});
			$(".loadingclass").modal("hide");
			

			$.each(data.sentences, function(iter, sentence){
				var subView = new App.RootRowView({model: sentence}); 
				$(".sentences").append(subView.render().el)
			})
			});

		//In case the jquery post request fails
		jqhr.fail(function(data){
				bootbox.alert(data.messege)
				self.$el.removeClass("dvLoading");
				event.stopPropagation();
		});

	
	},





	cancel: function(event){
		event.preventDefault();
		$(".data_selection").modal("hide");
		$.each($(":checkbox"), function(iter, __checkbox){
				$("#" + __checkbox.id).prop("checked", false); 
	})
	},

});


App.WordCloudWith_D3 = Backbone.View.extend({
	initialize: function(options){
		//In case svg was present on the page
		d3.select("svg").remove()
		this.model = options.model
		console.log(this.model)
		this._data = options.model
		this.ForceLayout(this.model);
		},
	

	dataFunction: function(value, LEVEL){

		function ADD_EX_BUBBLES(old_array){
			for(var i = 0; i < window.EX_BUBBLES; i++){
				old_array.push({"name": "NULL", 
						"polarity": 2, 
						"r": .1,})
			}
				    return old_array;
		}


		function if_data_empty(a_rray){
			if(a_rray == undefined){
				LEVEL = LEVEL -1
				bootbox.alert("There is no after level for this tag")
			}
		
			return true
		}
		
		var newDataSet = [];
		if (LEVEL == 0){
			$.each(this._data, function(i, __d){
				newDataSet.push({"name": __d.name, 
						"polarity": __d.polarity, 
						"positive": __d.positive,
						"negative": __d.negative,
						"neutral": __d.neutral,
						"sentences": __d.sentences,
						"similar": __d.similar,
						"o_likeness": __d.o_likeness,
						"i_likeness": __d.i_likeness,
						"r": __d.positive+__d.negative + __d.neutral,
						}); }); return newDataSet }
			
		if(LEVEL == 1){
			PARENT_LEVEL_1 = value
			$.each(this._data, function(i, __d){
				if (__d.name == value){
					$.each(__d.children, function(i, _d){
						newDataSet.push({"name": _d.name, 
							"polarity": _d.polarity, 
							"r": _d.frequency,
							}); }); }; }); return newDataSet}
		if(LEVEL == 2){ 
			PARENT_LEVEL_2 = value
			$.each(this._data, function(i, __d){
				if (__d.name == PARENT_LEVEL_1){
					$.each(__d.children, function(i, _d){
						if (_d.name == PARENT_LEVEL_2){
							if_data_empty(_d.children)
							$.each(_d.children, function(i, child){
								newDataSet.push({"name": child.name, 
									"polarity": child.polarity, 
									"r": child.frequency,
									}); }); }; }); }; }); return newDataSet}
		
		if(LEVEL == 3){ 
			PARENT_LEVEL_3 = value
			$.each(this._data, function(i, __d){
				if (__d.name == PARENT_LEVEL_1){
					$.each(__d.children, function(i, _d){
						if (_d.name == PARENT_LEVEL_2){
							$.each(_d.children, function(i, child){
								if (child.name == PARENT_LEVEL_3){
									if_data_empty(child.children)
									$.each(child.children, function(i, __child){
										newDataSet.push({"name": __child.name, 
										"polarity": __child.polarity, 
										"r": __child.frequency,
								}); }); }; }); }; }); }; }); return newDataSet}
		},


	ForceLayout: function(_data){
		/* The function to update svg elements if the window is being resized
		 function updateWindow(){
		 *     x = w.innerWidth || e.clientWidth || g.clientWidth;
		 *         y = w.innerHeight|| e.clientHeight|| g.clientHeight;
		 *
		 *             svg.attr("width", x).attr("height", y);
		 *             }
		 *             window.onresize = updateWindow;
		/* To Add scales to radius so that the nodes fit into the window
		 * http://alignedleft.com/tutorials/d3/scales
		 * consult the above mentioned tutorials
		 */

		_this = this;
		function DATA(value, LEVEL){return  _this.dataFunction(value, LEVEL)}
		LEVEL = 0
		var width = $(window).width() - 100;
		var height = $(window).height();

		var fill = d3.scale.category10();
		var color = d3.scale.category10().domain(d3.range(10000));


		var duration = 2000;
		var delay = 2;



		var force = d3.layout.force()
			.size([width, height])
			.charge(100)
		
		var svg = d3.select(".main-body").append("svg")
			.attr("width", width)
			.attr("height", height);

		var g = svg.append("g")
				.attr("transform", "translate(" + 0 + "," + 0 + ")")

		//This is the function wchich returns the maximum radius of the bubbles 
		function drawNodes(nodes){
			function convert_polarity(polarity){
				return polarity == 1? "positive":"negative" 

			};
		
			var RScale = d3.scale.linear()
				.domain(d3.extent(nodes, function(d) { return d.r; }))
			 	.range([width/25, 2*(width/25)])
		

			function rmax(){
				var RMAX = 0
				$.each(nodes, function(i, __node){
					if (RMAX < RScale(__node.r)){
						RMAX = RScale(__node.r)
					}
				});
				return RMAX
			}

			function tick(e){
				/*This prevents the bubbles to be dragged outside the svg element */
				node.attr("cx", function(d) { return d.x = Math.max(rmax(), Math.min(width - rmax(), d.x)); })
					        .attr("cy", function(d) { return d.y = Math.max(rmax(), Math.min(height - rmax(), d.y)); });
				
				
				var q = d3.geom.quadtree(nodes),
				i = 0,
				n = nodes.length;
				while (++i < n) q.visit(collide(nodes[i]));

					//.attr("cx", function(d) { return d.x; })
					//.attr("cy", function(d) { return d.y; })
					node
						.transition()
						.duration(75)
						.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
				};

			function collide(_node){
				var r = RScale(_node.r);
	
				nx1 = _node.x - r,
				nx2 = _node.x + r,
				ny1 = _node.y - r,
				ny2 = _node.y + r;
	    
				return function(quad, x1, y1, x2, y2){
					if (quad.point && (quad.point !== _node)){
						var x = _node.x - quad.point.x,
						y = _node.y - quad.point.y,
						l = Math.sqrt(x * x + y * y),
						r = RScale(_node.r) + RScale(quad.point.r);
						if (l < r){
							l = (l - r)/l*.5;
							_node.x -= x *= l;
							_node.y -= y *= l;
							quad.point.x += x;
							quad.point.y += y;
							}
					}
					return x1 > nx2 || x2 < nx1 || y1 > ny2 || y2 < ny1;};
			
					}
		function charge(d){return RScale(d.r) + 1 }     

		force.nodes(nodes)
			.gravity(.09)
			.charge(charge)
			.start()
		force.on("tick", tick)

		var node = g.selectAll(".node")
				.data(nodes, function(d) { return d.name; })
	
	
		
		node.enter()
			.append("g")
			.attr("class", "node")
			.on("dblclick", OnDBLClick)
			.attr("transform", function(d) { return "translate(" + d.x + "," + d.y + ")"; })
			.call(force.drag)
		
			
		node.append("circle")
			.attr("r", 0)
			.attr("fill", function(d, i){console.log(fill(parseInt(Math.random()*1000+ i))); return fill(parseInt(Math.random()*10+ i))}) 
			.attr("class", "node")
			//.attr("fill", function(d){return d.polarity ? "#66CCFF" : "#FF0033" }) 
			//.style("stroke", function(d, i) { return d3.rgb(fill(i & 3)).darker(10); })
			//.style("stroke", function(d, i) { return "#e377c2" })
			//.style("stroke", function(d, i) { return d.polarity == 1? "blue":"red" })
			//.style('stroke-width', '10px')
			.on("mousedown", function() { d3.event.stopPropagation(); })


		$('svg g').tipsy({ gravity: 'w', 
					html: true, 
					title: function(){
					//return  "<br>" + 'Name: ' +'  ' +'<span>' + this.__data__.name + '</span>' +"<br>" + 'Frequency: ' +  '<span>' + this.__data__.r + '</span>';}
					return   '<span>' + this.__data__.name + '</span>' + '<br>'+ '<span>' + 'Positive: ' + this.__data__.positive + '</span>' +  "<br>" + 'Negative: ' +  '<span>' + this.__data__.negative + '</span>' + '<span>' +  "<br>" + 'Neutral: ' +  '<span>'+ this.__data__.neutral + '</span>'+ '<br>'+ '<span>' + 'I_Likeness: '+ this.__data__.i_likeness + '</span>' + '<br>'+ '<span>'+ 'O_Likeness: ' + this.__data__.o_likeness + '</span>';}
				      });



		node.append('foreignObject')	
			.attr("class", "foreign")
			.append('xhtml:div')
			.style("font-size", 0)
			.append("p")	
		
		d3.selectAll("circle")
			.transition()
			//.duration(2000)
			.ease("cicle")
			.duration(1000) // this is 1s
			.delay(50)
			.attr("r", function(d){return RScale(d.r)})
			
		
		.transition()
		.each("end", function(){		
				d3.selectAll(document.getElementsByTagName("foreignObject")).transition()
					.attr('x', function(d){console.log(this.parentNode.getBBox()); return this.parentNode.getBBox().x/1.5})
					.attr('y', function(d){return this.parentNode.getBBox().y/2})
					.attr('width', function(d){ return 2*RScale(d.r) * Math.cos(Math.PI / 4)})
					.attr('height', function(d){ return 2*RScale(d.r) * Math.cos(Math.PI / 4)})
					.attr('color', 'black')
					.each(getSize)
			
			})

		.transition()
		.each("end", function(){		
				d3.selectAll(document.getElementsByTagName("foreignObject")).selectAll("div").transition()
					.style("font-size", function(d){return RScale(d.r)/5 + "px"})
			
			})	
			
		.transition()
		.each("end", function(){		
				d3.selectAll(document.getElementsByTagName("foreignObject")).selectAll("div").selectAll("p").transition()
					.text(function(d) { return d.name.substring(0, RScale(d.r) / 3)})
					.attr('id', "node-bubble")
					.style("text-align", "center")
					.style("vertical-align", "middle")
					.style("padding", "10px 5px 15px 20px")
					.style("line-height", "1")
			})
		

		node.exit().transition().attr("r", 0).duration(750).remove();
		
		
		d3.select("body")
			.on("mousedown", mousedown)
		
		function mousedown(){
			nodes.forEach(function(o, i){
				o.x += (Math.random() - .5) * 70;
				o.y += (Math.random() - .5) * 70;
			});
			force.resume();
		}
		
		}
		


		function getSize(d){
			var radius ;
			var bbox = this.getBBox();
			var cbbox = this.parentNode.getBBox(); 
			radius = this.parentNode.firstChild.getAttribute("r")
		}




		function OnDBLClick(d){
			/*
			LEVEL = LEVEL+1
			console.log(DATA(d.name, LEVEL))
			drawNodes(DATA(d.name, LEVEL))
			console.log(d)
			console.log(LEVEL)
			*/
			$(".sentences").empty()
			$.each(d.similar, function(iter, __data){
				__html = '<button class="btn btn-small btn-primary" type="button" style="margin: 10px">' + __data + '</button>';
				$('.sentences').append(__html)
			})
			$.each(d.sentences, function(iter, __sent){
				var subView = new App.RootRowView({model: __sent}); 
				$(".sentences").append(subView.render().el)

			})			
			console.log(d)
			console.log(d.similar)
			console.log(d.i_likeness)
			console.log(d.o_likeness)
		
		};

		console.log(DATA(null, LEVEL))
		drawNodes(DATA(null, LEVEL))
		//drawNodes(data())
	},



})



})

