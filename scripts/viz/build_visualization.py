import json
from pathlib import Path

def dict_to_array(node):
    if "children" in node and isinstance(node["children"], dict):
        node["children"] = [dict_to_array(v) for v in node["children"].values()]
    return node

def generate_visualization():
    leaf_path = Path("data/tree/leaf_nodes.json")
    if not leaf_path.exists():
        print(f"File not found: {leaf_path}")
        return

    with open(leaf_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build the tree structure dynamically from layer_labels
    tree = {"name": "Corpus", "children": {}}
    
    for leaf_id, leaf_info in data.items():
        labels = leaf_info.get("layer_labels", {})
        
        path = [
            f"Topic {labels.get('topic', 'N/A').replace('T', '')}",
            f"Emotion {labels.get('emotion', 'N/A').replace('E', '')}",
            f"Demographic {labels.get('demographic', 'N/A').replace('D', '')}",
            f"Register {labels.get('register', 'N/A').replace('R', '')}",
            f"Readability {labels.get('readability', 'N/A').replace('C', '')}"
        ]
        
        current = tree
        for i, node_name in enumerate(path):
            if "children" not in current:
                current["children"] = {}
                
            if node_name not in current["children"]:
                current["children"][node_name] = {"name": node_name}
            
            if i == len(path) - 1: # Last node (leaf)
                current["children"][node_name]["value"] = leaf_info.get("document_count", 1)
            
            current = current["children"][node_name]

    echarts_data = [dict_to_array(tree)]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hierarchical Stratification Tree Visualization</title>
    <!-- Include ECharts -->
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: Arial, sans-serif;
            background-color: #fafafa;
        }}
        #main {{
            width: 100%;
            height: 100vh;
        }}
        .header {{
            position: absolute;
            top: 10px;
            left: 20px;
            z-index: 100;
            background: rgba(255,255,255,0.9);
            padding: 10px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        h1 {{ margin: 0 0 5px 0; font-size: 20px; color: #333; }}
        p {{ margin: 0; font-size: 14px; color: #666; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Hierarchical Corpus Tree</h1>
        <p>Topic ➔ Emotion ➔ Demographic ➔ Register ➔ Readability</p>
    </div>
    <div id="main"></div>

    <script>
        var chartDom = document.getElementById('main');
        var myChart = echarts.init(chartDom);
        var option;

        var data = {json.dumps(echarts_data)};

        option = {{
            tooltip: {{
                trigger: 'item',
                formatter: '{{b}}: {{c}} chunks'
            }},
            series: {{
                type: 'sunburst',
                data: data,
                radius: [0, '95%'],
                sort: 'desc',
                emphasis: {{
                    focus: 'ancestor'
                }},
                levels: [
                    {{}}, // Blank root
                    {{ // Topic Level
                        r0: '10%',
                        r: '25%',
                        itemStyle: {{ borderWidth: 1 }},
                        label: {{ 
                            position: 'outside',
                            silent: false,
                            minAngle: 5
                        }}
                    }},
                    {{ // Emotion Level
                        r0: '25%',
                        r: '45%',
                        label: {{ align: 'right', minAngle: 5 }}
                    }},
                    {{ // Demographic Level
                        r0: '45%',
                        r: '65%',
                        label: {{ align: 'right', minAngle: 5 }}
                    }},
                    {{ // Register Level
                        r0: '65%',
                        r: '80%',
                        label: {{ align: 'right', minAngle: 5 }}
                    }},
                    {{ // Readability Level
                        r0: '80%',
                        r: '95%',
                        itemStyle: {{ color: '#aaa', borderWidth: 0.5 }},
                        label: {{ position: 'outside', minAngle: 10, padding: 3, silent: false }}
                    }}
                ]
            }}
        }};

        myChart.setOption(option);
        
        window.addEventListener('resize', function() {{
            myChart.resize();
        }});
    </script>
</body>
</html>
"""

    out_path = Path("tree_visualization.html")
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Successfully generated independent interactive visualization at {out_path.absolute()}")

if __name__ == "__main__":
    generate_visualization()
