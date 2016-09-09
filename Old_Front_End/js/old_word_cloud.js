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
		var jqhr = $.post(window.get_start_date_for_restaurant, {"eatery_id": $("#eateriesList").find('option:selected').attr("id")})	
		jqhr.done(function(data){
			console.log(data.result)
			self.$("#startDate").val(data.result.start)
			self.$("#selectStartDate").val(data.result.start)
			self.$("#endDate").val(data.result.end)
			self.$("#selectEndDate").val(data.result.end)
		});


	}, 

	events: {
		"click #idSubmit": "submit",
		"click #idCancel": "cancel",

	},

	loading_bootbox: function(){
		$(".data_selection").modal("hide");
		bootbox.dialog({ 
			closeButton: false, 
			message: "<img src='css/images/loading_3.gif'>",
			className: "loadingclass",
		});
	},

	submit: function(event){
		var self = this;
		$(".dynamic_display_word_cloud").empty();	
		event.preventDefault();
		this.$el.addClass("dvLoading");
	
		this.loading_bootbox()


		if ($('#startDate').val() > $('#selectStartDate').val()){
			bootbox.alert("Genius the start date selected should be greater then start date").find('.modal-content').addClass("bootbox-modal-custom-class");
			return

		}
		if ($('#endDate').val() < $('#selectEndDate').val()){

			bootbox.alert("Genius the end date selected should be less then end date").find('.modal-content').addClass("bootbox-modal-custom-class");
			return
		}
	
	
		var jqhr = $.post(window.get_word_cloud, {"eatery_id": $("#eateriesList").find('option:selected').attr("id"),
							"start_date": $('#selectStartDate').val(),
							"end_date": $('#selectEndDate').val(),
		    					"category": $("#wordCloudCategory").find("option:selected").val(),
						})	
		
		
		//On success of the jquery post request
		jqhr.done(function(data){
			var subView = new App.WordCloudWith_D3({model: data.result});
			$(".loadingclass").modal("hide");
			});

		//In case the jquery post request fails
		jqhr.fail(function(){
				bootbox.alert("Either the api or internet connection is not working, Try again later")
				self.$el.removeClass("dvLoading");
				event.stopPropagation();
		});

	
	},





	cancel: function(event){
		event.preventDefault();
		$(".data_selection").modal("hide");
	},

});

App.WordCloudWith_D3 = Backbone.View.extend({
	initialize: function(options){
		this.model = options.model;
		console.log(this.model)
		$.each(this.model, function(i, __d){
			console.log(__d.name, __d.frequency, __d.polarity)
		
		})
	
		this.render();
	},


	render: function(){
		//copied from : https://github.com/vlandham/bubble_cloud/blob/gh-pages/coffee/vis.coffee
		//explanation at:

		var Bubbles, root, texts;		
		root = typeof exports !== "undefined" && exports !== null ? exports : this;
		function Bubbles(){
			var  chart, clear, click, collide, collisionPadding, connectEvents, data, force, gravity, hashchange, height, idValue, jitter, label, margin, maxRadius, minCollisionRadius, mouseout, mouseover, node, rScale, rValue, textValue, tick, transformData, update, updateActive, updateLabels, updateNodes, width
			width = $(window).width() - 50;
			height = $(window).height()*1.1;
			data = [];
			node = null;
			label = null;
			margin = {top: 5, right: 0, bottom: 0, left: 0};
			maxRadius = 80;
	
			rScale = d3.scale.sqrt().range([0,maxRadius])
	
				
			var tip = d3.tip()
				.attr('class', 'd3-tip')
				.offset([-10, 0])
				.html(function(d) {
					return "<strong style='color:black'>Frequency:</strong> <span style='color:red'>" + d.frequency + "</span>";
			    })

	
			function rValue(d){
				return parseInt(d.frequency);
				};

			function idValue(d){
				return d.name
				};
	
			function textValue(d){
				return d.name
					};

			collisionPadding = 4
			minCollisionRadius = 20
	
			jitter = 0.5;

			transformData = function(rawData){
				rawData.forEach(function(d){
				d.count = parseInt(d.count);
				return rawData.sort(function(){
					return 0.5 - Math.random();
				});
				})
				return rawData;
					};
		

			function tick(e){
				node.attr("cx", function(d) { return d.x = Math.max(rScale(rValue(d)), Math.min(width - 50, d.x)); })
					.attr("cy", function(d) { return d.y = Math.max(rScale(rValue(d)), Math.min(height - 50, d.y)); });
				
				var dampenedAlpha;
				dampenedAlpha = e.alpha * 0.1

				node.each(gravity(dampenedAlpha))
				.each(collide(jitter))
				.attr("transform", function(d){
					return "translate(" + d.x + "," + d.y + ")";
				})


				texts.style("left", function(d) { return (margin.left + d.x) - d.dx / 2})
				.style("top", function(d){(margin.top + d.y) - d.dy / 2})
			};
	
	
	
			force = d3.layout.force()
					.gravity(0)
					.charge(0)
					.size([width, height])
					.on("tick", tick)
			

	
	
			rawData = this.model
			
			function chart(selection){
				return selection
					.attr("id", "background_class")
					.each(function(rawData){
					var maxDomainValue, svg, svgEnter;
					data = transformData(rawData)
					maxDomainValue = d3.max(data, function(d){ return rValue(d)})
					rScale.domain([0, maxDomainValue])
		
					svg = d3.select(this)
						.selectAll("svg")
						.data([data])
				
					svgEnter = svg.enter().append("svg");
					svg.attr("width", width + margin.left + margin.right);
					svg.attr("height", height + margin.top + margin.bottom);	      

					svg.call(tip)	
					
					
					//This Function will add shadow to the nodes
					addShadow(svg)	
					
					node = svgEnter.append("g")
						.attr("id", "bubble-nodes")
						.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

					node.append("rect")
						.attr("id", "bubble-background")
						.attr("width", width)
						.attr("height", height)

					update()
				})
			};


		 function update(){
			data.forEach(function(d, i){
				return d.forceR = Math.max(minCollisionRadius, rScale(rValue(d)));
			});
			force.nodes(data).start()

			updateNodes()
				};

		function getSize(d) {
			var radius ;
			var bbox = this.getBBox();
			cbbox = this.parentNode.getBBox();
			radius = this.parentNode.firstChild.getAttribute("r")	
			scale = radius/3;
			d.scale = scale;
		}


		function updateNodes(){
			node = node.selectAll(".bubble-node").data(data, function(d){ return idValue(d)})

			node.exit().remove()
			
			node.enter()
				.append("a")
				.style("filter", "url(#drop-shadow)")
				.attr("class", "bubble-node")
				.attr("fill", function(d){return d.polarity ? "#66CCFF" : "#FF0033" })
				.call(force.drag)
				.call(connectEvents)
				.append("circle")
				.attr("r", function(d){ return rScale(rValue(d))
				})
			
			//Adding tooltip
			$('svg circle').tipsy({ 
					        gravity: 'w', 
					        html: true, 
					        title: function(){
							return 'Frequency: ' + '<span>' + this.__data__.frequency + '</span>';
						}
				      });



			texts = node.append("text")
					      .style("text-anchor", "middle")
					      .attr("dy", ".3em")
					      .attr('fill', 'black')
					      .attr("label-name", function(d) { return textValue(d)})
					      .text(function(d) { return textValue(d)})
					      .each(getSize)
					      //.style("font-size", function(d){ return Math.max(1, rScale(rValue(d)/2))+"px"})
					      .style("font-size", function(d){ return d.scale+"px"})
					      .style("width", function(d){ return rScale(rValue(d))+"px"})
		}

			function addShadow(svg){
					defs = svg.append("defs");
					filter = defs.append("filter")
						    .attr("id", "drop-shadow")
						    .attr("height", "150%")
						    .attr("width", "200%")
					filter.append("feGaussianBlur")
						.attr("in", "SourceAlpha")
						.attr("stdDeviation", 5)
						.attr("result", "blur");

					feOffset = filter.append("feOffset")
						    .attr("in", "blur")
						    .attr("dx", 5)
						    .attr("dy", 5)
						    .attr("result", "offsetBlur");
					feMerge = filter.append("feMerge");
							feMerge.append("feMergeNode")
							.attr("in", "offsetBlur")
					
					feMerge.append("feMergeNode")
						.attr("in", "SourceGraphic");

				}
		
	
	

		function gravity(alpha){
			cx = width/2
			cy = height/2
			ax = alpha/8
			ay = alpha

			return function(d){
				d.x += (cx - d.x)*ax
				return d.y += (cy - d.y)*ay
				};
		};


		function collide(jitter){
			return function(d){
				return data.forEach(function(d2){
					var distance, minDistance, moveX, moveY, x, y;
					
					if (d != d2){
						x = d.x - d2.x
						y = d.y - d2.y
						distance = Math.sqrt(x*x+y*y)
						minDistance = d.forceR + d2.forceR + collisionPadding
          
						if (distance < minDistance){
							distance = (distance-minDistance)/distance*jitter
							moveX = x*distance
							moveY = y*distance
							d.x -= moveX
							d.y -= moveY
							d2.x += moveX
							d2.y += moveY
						}
					}
						})
				}
					}


		function connectEvents(d){
		//	d.on("click", onclick)
		};

		function onclick(d){
			console.log("Click on the bubble has been initiated")
			event.preventDefault();
			function increaseRadius(selector){
				d3.select(selector).select("circle")
					.attr("class", "clicked_bubble")
					.transition()
					.duration(7500)
					.attr("r", rScale(rValue(d))*3)
			}


			function increaseTextSize(selector){
				d3.select(selector).select("text")
					.attr("class", "clicked_bubble")
					.transition()
					.duration(7500)
					.style("font-size", function(d){ return d.scale*3+"px"});
				}



			console.log(d.frequency)
			//location.replace("#" + encodeURIComponent(idValue(d)))
			
			d3.transition()
				.ease("linear")
					.each(function() {
						d3.selectAll(".bubble-node").transition()
						.duration(7500)
				              .style("opacity", function(){ 
						      if (this.childNodes[1].getAttribute("label-name") != d.name)
						{
							return 0;
						}
						      
					      }
						      
						      )
			        })
			
					.transition()
					.ease("linear")
					.call(increaseRadius(this))

		}		    
		 tooltip = d3.select("body")
			      .append("div")
			          .style("position", "absolute")
				      .style("z-index", "10")
				          .style("visibility", "hidden")
					      .text("a simple tooltip");



		function mouseover(d){
			console.log(d)
			//.tooltip({ content: "Awesome title!" });
			tooltip.text(d.name); 
			  return tooltip.style("visibility", "visible");
				/*
			return node.classed("bubble-hover", function(p){
				console.log()
				return p === d
				;
			});
			*/
			      };
		function mouseout(d){
			return node.classed("bubble-hover", false);
			        };

		  chart.jitter = function(_) {
			      if (!arguments.length) {
				            return jitter;
					        }
			          jitter = _;
				      force.start();
				          return chart;
					    };
		    chart.height = function(_) {
			        if (!arguments.length) {
					      return height;
					          }
				    height = _;
				        return chart;
					  };
		      chart.width = function(_) {
			          if (!arguments.length) {
					        return width;
						    }
				      width = _;
				          return chart;
					    };
		        chart.r = function(_) {
				    if (!arguments.length) {
					          return rValue;
						      }
				        rValue = _;
					    return chart;
					      }


		return chart

		}
		
		plotData = function(selector, data, plot){
			return d3.select(selector).datum(data).call(plot);
		};
		 
		
		plot = Bubbles();

		plotData(".dynamic_display", this.model, plot);

	},
		
})

})

