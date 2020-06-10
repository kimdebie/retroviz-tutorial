var svg;
var legend_svg;

// set size and margins of plot
var margin = {top: 30, right: 10, bottom: 10, left: 0},
    width = 900 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// elements taken from json
var points = current_pt.points;
var ranges = current_pt.ranges;
var retro_score = current_pt.retro_score;
var target = current_pt.target;
var dimensions = Array.from(current_pt.sorted_axes);

// target variable added as dimension
dimensions.push(target);

// build a linear scale for each dimension
var y = {}
for (i in dimensions) {
  name = dimensions[i]
  y[name] = d3.scaleLinear()
    .domain([d3.min(ranges, function(d) { return d[name] }),
            d3.max(ranges, function(d) { return d[name] })])
    .range([height, 0]);
}

// scale finds position for axis
x = d3.scalePoint()
      .range([0, width])
      .padding(1)
      .domain(dimensions);

// path function takes datapoint and returns coordinates
function path(d) {
  return d3.line()(dimensions.map(function(p) { return [x(p), y[p](d[p])]; }));
}

// color scale for different colors of new instance and neighbors
var color = d3.scaleOrdinal()
              .domain(["new_point", "neighbor"])
              .range(["#f84b26", "#8aa7f3"])

// stroke wider for new point and narrower for neighbors
var strokewidth = d3.scaleOrdinal()
                    .domain(["new_point", "neighbor"])
                    .range([3, 1.5])

// opacity is higher for new point than for neighbors
var opacity = d3.scaleOrdinal()
                .domain(["new_point", "neighbor"])
                .range([1, 0.5])

// function to set fontweight bold for target variable
var fontweight = function(d) {
  if(d===target) {
    return "bold"
  } else {
    return ""
  }
}

// font size larger for target variable
var fontsize = function(d) {
  if(d===target) {
    return "larger"
  } else {
    return "inherit"
  }
}

// highlight line that is hovered over
var highlight = function(d, i){
    var selected_point = d.point_type + i;
    var pointtype = d.point_type;

    // all lines reduce opacity
    d3.selectAll(".line")
      .transition().duration(150)
      .style("opacity", "0.2")

    // hovered line gets higher opacity
    d3.select("#" + selected_point)
      .transition().duration(150)
      .style("opacity", "1")

    // show value on axis
    d3.selectAll(".axistitle")
        .text(function(d) { var id = d3.select(this).attr("id"); return id + ": " + points[i][id].toFixed(1) } );

    // tooltip hover
    var tooltipname = "#" + d.point_type + i;
    d3.selectAll(tooltipname).style("opacity", 1);
}

// unhighlight line that was hovered over (back to normal state)
var unhighlight = function(d, i) {

  d3.selectAll(".line")
    .transition().duration(150)
    .style("opacity", function(d) { return opacity(d.point_type) } )

  d3.selectAll(".axistitle")
    .text(function(d) { var id = d3.select(this).attr("id"); return id })

    // tooltip hover
  var tooltipname = "#" + d.point_type + i;
  d3.selectAll(tooltipname).style("opacity", 0);
}

// setting the width of the legend
var legend_width = 300;

// color scale for legend
var legend_colorscale = d3.scaleLinear()
                          .domain([0, 0.5, 1])
                          .range(["#f63528", "#ffffff", "#30e371"]);

// position of line on legend
var legend_linescale = d3.scaleLinear()
                          .domain([0,1])
                          .range([0, legend_width])

// create svg object which will contain plot
svg = d3.select("#pcp")
        .append("svg")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
          .attr("id", "pcpplot")
        .append("g")
          .attr("transform",
                "translate(" + margin.left + "," + margin.top + ")");

// draw axis
svg.selectAll("axis")
    .data(dimensions).enter()
    .append("g").attr("class", "axes")
    .attr("transform", function(d) { return "translate(" + x(d) + ")"; })
    .each(function(d) { d3.select(this).call(d3.axisLeft().scale(y[d])); })
    .append("text")
      .style("text-anchor", "middle")
      .attr("y", -9)
      .attr("class", "axistitle")
      .attr("id", function(d) { return d; })
      .text(function(d) { return d; })
      .style("fill", "black")
      .style("font-weight", function(d) { return fontweight(d) } )
      .style("font-size", function(d) { return fontsize(d) } );

// add lines
svg.selectAll(".datalines")
    .data(points)
    .enter().append("path")
      .attr("d",  path)
      .attr("class", "line")
      .attr("id", function (d,i) { return d.point_type + i})
      .style("fill", "none")
      .style("stroke", function(d) { return color(d.point_type) } )
      .style("stroke-width", function(d) { return strokewidth(d.point_type) } )
      .style("opacity", function(d) { return opacity(d.point_type) } )
      .on("mouseover", highlight)
      .on("mouseout", unhighlight);

// color scale for legend and retro score.
var legend_colorscale = d3.scaleLinear()
                          .domain([0, 0.5, 1])
                          .range(["#f63528", "#ffffff", "#30e371"]);

// create svg which will contain legend
legend_svg = d3.select("#legend_ts")
                  .append("svg")
                  .attr("width", 300)
                  .attr("height", 50)
                  .attr("id", "legendplot");

// other legend elements
var defs = legend_svg.append("defs");

var linearGradient = defs.append("linearGradient")
                        .attr("id", "linear-gradient");

linearGradient.selectAll("stop")
              .data(legend_colorscale.range())
              .enter().append("stop")
              .attr("offset", function(d,i) { return i/(legend_colorscale.range().length-1); })
              .attr("stop-color", function(d) { return d; });

// creates the gradient
legend_svg.append("rect")
          .attr("width", legend_width)
          .attr("height", 30)
          .attr("x", 0)
          .attr("y", 10)
          .attr("id", "legendrect")
          .style("fill", "url(#linear-gradient)");

// place vertical line on legend
legend_svg.append("line")
          .attr("id", "legendline")
          .attr("x1", legend_linescale(retro_score))
          .attr("y1", 5)
          .attr("x2", legend_linescale(retro_score))
          .attr("y2", 45)
          .attr("stroke-width", 2)
          .attr("stroke", "black")

// place a 0 on legend
legend_svg.append("text")
          .attr("class", "legendtext")
          .attr("x", 3)
          .attr("y", 30)
          .text("0");

// place a 1 on legend
legend_svg.append("text")
          .attr("class", "legendtext")
          .attr("x", 290)
          .attr("y", 30)
          .text("1");

var shadeTarget = function(d) {
  if (d === target) {
    return 0.15;
  } else {
    return 0;
  }
}

var shadeTargetName = function(d) {
  if (d === target) {
    return 1;
  } else {
    return 0;
  }
}

var targetName = function(d) {
  if (d === target) {
    return "target variable";
  } else {
    return "";
  }
}

var barwidth = 120;
var rectWidth = function(d) {
  if (d === target) {
    return barwidth;
  } else {
    return 0;
  }
}

var showTarget = function(d) {

  if (d === target) {
    d3.select(".targetname + #" + target)
      .style("opacity", 1)
  }
}

var hideTarget = function(d) {
  //console.log(d3.selectAll(".targetname"));

  if (d === target) {
    d3.select(".targetname + #" + target)
      .style("opacity", 0)
  }
}

// draw box around last axis
svg.selectAll("rect")
   .data(dimensions).enter()
   .append("rect")
   .attr("x", function(d) { return x(d) - barwidth/2 })
   .attr("y", -25)
   .attr("width", function(d) { return rectWidth(d) })
   .attr("height", height + margin.top + margin.bottom)
   .style("fill", "grey")
   .style("opacity", function(d) { return shadeTarget(d) ; })
   .on("mouseover", showTarget)
   .on("mouseout", hideTarget);

// text on hover on last axis
svg.selectAll(".targetname")
   .data(dimensions).enter()
   .append("text")
   .attr("class", "targetname")
   .attr("id", function(d) { return d })
   .attr("transform", function(d) { return "translate(" + (x(d) + 20) + ",220) rotate(270)" })
   .text(function(d) { return targetName(d) })
   .style("opacity", 0);

// tooltip on line (neighbor or other point)
svg.selectAll(".linetooltip")
  .data(points).enter()
  .append('text')
  .html(d => d.point_type)
  .attr("class", "linetooltip")
  .attr("id", function(d, i) { return d.point_type + i } )
  .attr('fill', "black")
  .attr("opacity", 0)
  .attr('alignment-baseline', 'middle')
  .attr('x', x(target))
  .attr('dx', '.5em')
  .attr('y', d => y[target](d[target]));

// add text
var par = d3.select("#description1").append("p");
par.append("span").text("The RETRO-score for this prediction is ");
par.append("strong").text(retro_score.toFixed(3))
