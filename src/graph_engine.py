import json
import re

class LightweightGraph:
    def __init__(self):
        self.nodes = {} # {entity: {type: ..., data: ...}}
        self.edges = [] # [(source, target, relationship)]

    def extract_entities(self, text):
        """
        使用正则和规则提取简单的财务实体
        """
        entities = []
        # 1. 提取百分比变化
        pct_matches = re.findall(r'(\w+)(?:增长|下降|变动)(?:了|比例为)?([\d\.]+%|[\d\.]+个百分点)', text)
        for metric, val in pct_matches:
            entities.append({"name": metric, "type": "Metric", "val": val})
        
        # 2. 提取公司/组织
        org_matches = re.findall(r'([^\s，。]{2,20}(?:有限公司|集团|分公司|子公司))', text)
        for org in org_matches:
            entities.append({"name": org, "type": "Organization"})
            
        return entities

    def build_from_chunks(self, chunks_data):
        print(f"🕸️ 正在从 {len(chunks_data)} 个文本块中构建知识图谱...")
        for chunk in chunks_data:
            content = chunk["content"]
            entities = self.extract_entities(content)
            
            for ent in entities:
                name = ent["name"]
                if name not in self.nodes:
                    self.nodes[name] = {"type": ent["type"], "occurrences": 1}
                else:
                    self.nodes[name]["occurrences"] += 1
        
        print(f"✅ 图谱构建完成：发现 {len(self.nodes)} 个核心实体。")

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"nodes": self.nodes, "edges": self.edges}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # 简单自测
    graph = LightweightGraph()
    test_text = "比亚迪股份有限公司控股子公司比亚迪半导体增长了15%。"
    print(graph.extract_entities(test_text))
