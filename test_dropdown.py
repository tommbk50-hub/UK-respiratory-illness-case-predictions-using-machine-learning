with open("sars-cov-2_homepage.html") as f:
    content = f.read()
    if 'dropup' in content:
        print("dropup found!")
    else:
        print("no dropup class found")
