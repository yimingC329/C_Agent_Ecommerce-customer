import os
from pathlib import Path as _ConfigPath
from dotenv import load_dotenv as _load_local_env
for _config_parent in _ConfigPath(__file__).resolve().parents:
    if _config_parent.name == "llm_backend":
        _load_local_env(_config_parent / ".env", override=False)
        break
from neo4j import GraphDatabase





NEO4J_URI = os.environ['NEO4J_URL']
NEO4J_USERNAME = os.environ['NEO4J_USERNAME']
NEO4J_PASSWORD = os.environ['NEO4J_PASSWORD']
NEO4J_DATABASE="neo4j"         # 社区版仅能使用默认的neo4j数据库，不支持创建其他数据库



driver = GraphDatabase.driver(

    NEO4J_URI, 

    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)

    )



driver = GraphDatabase.driver(

    NEO4J_URI, 

    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)

    )



def test_connection():

    # 在这里创建会话

    with driver.session() as session:

        session.run("MATCH (n) RETURN n LIMIT 1")



try:

    test_connection()

    print("连接成功！")

except Exception as e:

    print("连接失败:", e)

finally:

    driver.close()  # 确保在所有操作完成后再关闭驱动程序