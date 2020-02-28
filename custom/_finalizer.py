import matplotlib.pyplot as plt
import pandas as pd

def finalizer(res):

    pages = res["scatter"]
    pages = list(map(lambda x: x["user_meta"], pages))
    df = pd.DataFrame(pages)
    #df.to_csv('./result/res.csv')

    corr_df = df.loc[:, ["time", "n_howtomake", "n_components"]].corr()
    #corr_df.to_csv('./result/corr.csv')

    res["analyzed"] = {}
    res["analyzed"]["correlation"] = {}
    res["analyzed"]["correlation"]["time-n_howtomake"] = corr_df.loc["time", "n_howtomake"]
    res["analyzed"]["correlation"]["time-n_components"] = corr_df.loc["time", "n_components"]
    res["analyzed"]["correlation"]["n_howtomake-n_components"] = corr_df.loc["n_howtomake", "n_components"]

    fig = plt.figure()
    x = df["time"]
    y = df["n_howtomake"]
    plt.scatter(x, y)
    plt.savefig('./result/time-n_howtomake.png')

    fig = plt.figure()
    x = df["time"]
    y = df["n_components"]
    plt.scatter(x, y)
    plt.savefig('./result/time-n_components.png')

    fig = plt.figure()
    x = df["n_howtomake"]
    y = df["n_components"]
    plt.scatter(x, y)
    plt.savefig('./result/n_howtomake-n_components.png')

    del res["categorized_doc"]
    del res["num_topics"]
    del res["perplexity"]
    del res["topic_details"]
    del res["topic_labels"]

    for l in range(len(res["scatter"])):
        res["scatter"][l].update(**res["scatter"][l]["user_meta"])
        del res["scatter"][l]["user_meta"]
        del res["scatter"][l]["topic_id"]
        #del res["scatter"][l]["x"]
        #del res["scatter"][l]["y"]

    res['pages'] = res.pop('scatter')
    
    return res
