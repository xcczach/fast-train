from ddgs import DDGS

with DDGS() as ddgs:
    results = ddgs.text("Python 并发编程", max_results=5)
    for r in results:
        print(r["title"], r["href"], r.get("body"))
