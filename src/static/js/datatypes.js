d3.text("../static/samples/datatypes.csv").then(function(datasetText) {
                  var rows  = d3.csvParseRows(datasetText),
                      table = d3.select('#datatype').append('table')
                                .style("border-collapse", "collapse")
                                .style("border", "2px black solid");

                  // headers
                  table.append("thead").append("tr")
                    .selectAll("th")
                    .data(rows[0])
                    .enter().append("th")
                    .text(function(d) { return d; })
                    .style("border", "1px black solid")
                    .style("padding", "5px")
                    .style("background-color", "lightgray")
                    .style("font-weight", "bold")
                    .style("text-transform", "uppercase");

                  // data
                  table.append("tbody")
                    .selectAll("tr").data(rows.slice(1))
                    .enter().append("tr")
                    .selectAll("td")
                    .data(function(d){return d;})
                    .enter().append("td")
                    .style("border", "1px black solid")
                    .style("padding", "5px")
                    .on("mouseover", function(){
                    d3.select(this).style("background-color", "powderblue");
                  })
                    .on("mouseout", function(){
                    d3.select(this).style("background-color", "white");
                  })
                    .text(function(d){return d;})
                    .style("font-size", "12px");
            });


