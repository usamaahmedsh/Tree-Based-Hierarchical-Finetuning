import json
from pathlib import Path

def dict_to_array(node):
    if "children" in node and isinstance(node["children"], dict):
        node["children"] = [dict_to_array(v) for v in node["children"].values()]
    return node

def generate_visualization():
    leaf_path = Path("data/tree/leaf_nodes.json")
    if not leaf_path.exists():
        print("Leaf nodes file not found")
        return

    with open(leaf_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tree = {"name": "Corpus", "children": {}}
    
    for leaf_id, leaf_info in data.items():
        labels = leaf_info.get("layer_labels", {})
        
        path = [
            f"Topic {labels.get('topic', '')}",
            f"Emotion {labels.get('emotion', '')}",
            f"Demographic {labels.get('demographic', '')}",
            f"Register {labels.get('register', '')}",
            f"Readability {labels.get('readability', '')}"
        ]
        
        current = tree
        for i, node_name in enumerate(path):
            if "children" not in current:
                current["children"] = {}
                
            if node_name not in current["children"]:
                current["children"][node_name] = {"name": node_name}
            
            if i == len(path) - 1: # Last node
                current["children"][node_name]["value"] = leaf_info.get("document_count", 1)
            
            current = current["children"][node_name]

    echarts_data = [dict_to_array(tree)]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Collapsible Tree Visualization</title>
    <!-- Include ECharts -->
    <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: Arial, sans-serif; }}
        #main {{ width: 100%; height: 100vh; }}
        .header {{ position: absolute; top: 10px; left: 20px; z-index: 100; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    </style>
</head>
<body>
    <div class="header">
        <h2>Collapsible Tree</h2>
        <p>Zoom and pan. Click nodes to collapse/expand.</p>
    </div>
    <div id="main"></div>
    <script>
        var chartDom = document.getElementById('main');
        var myChart = echarts.init(chartDom);
        var data = {json.dumps(echarts_data[0])};

        var option = {{
            tooltip: {{
                trigger: 'item',
                triggerOn: 'mousemove',
                formatter: function (info) {{
                    var value = info.value;
                    if (!value && info.treeAncestors) {{
                         // sum up values if it's a parent node
                         // Optional, basic tooltip
                    }}
                    return info.name + (value ? ': ' + value + ' chunks' : '');
                }}
            }},
            series: [
                {{
                    type: 'tree',
                    data: [data],
                    top: '5%',
                    left: '7%',
                    bottom: '5%',
                    right: '20%',
                    symbolSize: 7,
                    initialTreeDepth: 2,
                    label: {{
                        position: 'left',
                        verticalAlign: 'middle',
                        align: 'right',
                        fontSize: 12
                    }},
                    leaves: {{
                        label: {{
                            position: 'right',
                            verticalAlign: 'middle',
                            align: 'left'
                        }}
                    }},
                    emphasis: {{ focus: 'descendant' }},
                    expandAndCollapse: true,
                    animationDuration: 550,
                    animationDurationUpdate: 750
                }}
            ]
        }};
        myChart.setOption(option);
        window.addEventListener('resize', function() {{ myChart.resize(); }});
    </script>
</body>
</html>
"""

    out_path = Path("tree_visualization_collapsible.html")
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Successfully generated independent interactive visualization at {out_path.absolute()}")

if __name__ == "__main__":
    generate_visualization()
