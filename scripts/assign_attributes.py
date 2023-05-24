import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import networkx as nx
import random
from scipy.stats import poisson
import logging, sys

logging.disable(sys.maxsize)
import multiprocessing as mp
import functools
import operator
import ConfigModel_MCMC as CM
from scripts.rkernel_estimate import r_sample_distb
from scripts.subsimulation import sub_simulation
import time

urban_gradient = {1: "urban", 2: "suburban", 3: "rural"}
district = {1: "Buchanan_Urban", 2: "Buchanan_Suburban", 3: "Buchanan_Rural", 4: "Riley_Urban", 5: "Riley_Suburban",
            6: "Riley_Rural",
            7: "Platte_Urban", 8: "Platte_Suburban", 9: "Platte_Rural"}


# Data Cleaning
def clean_data(data, columns):
    """
    Drop outliers and missing values from code. This removes number above 100 and drops missing value represented by -99
    :param data:
    :param columns:
    :return:
    """
    data = data[data[columns] <= 100]
    data_cleaned_col = data[data[columns] > -1]
    return data_cleaned_col


def trust_filter(trust):
    # Removed fractions by conveerting to 0
    trust = [round(i) for i in trust]
    trust = list(filter(lambda i: i >= 0, trust))
    trust = list(filter(lambda i: i <= 10, trust))
    return trust


def source_filter(sources):
    sources = list(filter(lambda i: i >= 1, sources))
    sources = list(filter(lambda i: i <= 5, sources))
    return sources


def collect_users():
    survey_data = pd.read_csv(r"data/initial/Data_07152021.csv", encoding='cp1252')
    people_interactions = clean_data(survey_data, "Q4_ITR_people")
    return people_interactions


def group_data_urban_gradient(data):
    groups = data.groupby(["Urban_Gradient"])
    group_urban = groups.get_group(1)
    group_suburban = groups.get_group(2)
    group_rural = groups.get_group(3)
    return group_urban, group_suburban, group_rural


def group_data_district(data):
    grouped = data.groupby(["District_number"])
    buchanan_urban_grp = grouped.get_group(1)
    buchanan_suburban_grp = grouped.get_group(2)
    buchanan_rural_grp = grouped.get_group(3)

    riley_urban_grp = grouped.get_group(4)
    riley_suburban_grp = grouped.get_group(5)
    riley_rural_grp = grouped.get_group(6)

    platte_urban_grp = grouped.get_group(7)
    platte_suburban_grp = grouped.get_group(8)
    platte_rural_grp = grouped.get_group(9)

    return buchanan_urban_grp, buchanan_suburban_grp, buchanan_rural_grp, riley_urban_grp, riley_suburban_grp, \
           riley_rural_grp, platte_urban_grp, platte_suburban_grp, platte_rural_grp


all_data = collect_users()

# District
_, grouped1_buchanan_suburban, grouped1_buchanan_rural, grouped1_riley_urban, grouped1_riley_suburban, \
grouped1_riley_rural, grouped1_platte_urban, grouped1_platte_suburban, _ = group_data_district(all_data)

# Urban Gradient
grouped2_urban, grouped2_suburban, grouped2_rural = group_data_urban_gradient(all_data)


def configuration_model(node_deg):
    """
    Uses configuration model to create a graph
    :param node_deg:
    :return:
    """
    if sum(node_deg) % 2 != 0:
        node_deg[-1] += 1
    G1 = nx.configuration_model(node_deg)
    allow_loops = False
    allow_multi = False
    is_vertex_labeled = True
    mcmc_object = CM.MCMC(G1, allow_loops, allow_multi, is_vertex_labeled)
    G2 = mcmc_object.get_graph()
    return G2


# Learn from each sources of news by district and area
# Rewrite code with dicts in function definition

#For no disaster
def nodisaster_lcnews_nw():
    """
    This collects the frequency of interactions of each user with local news when there is no disaster
    :return:
    """

    urban = grouped2_urban['Q13_NM_sourcefrq_lcnews']
    suburban = grouped2_suburban['Q13_NM_sourcefrq_lcnews']
    rural = grouped2_rural['Q13_NM_sourcefrq_lcnews']

    buch_suburban = grouped1_buchanan_suburban['Q13_NM_sourcefrq_lcnews']
    buch_rural = grouped1_buchanan_rural['Q13_NM_sourcefrq_lcnews']
    ril_urban = grouped1_riley_urban['Q13_NM_sourcefrq_lcnews']
    ril_suburban = grouped1_riley_suburban['Q13_NM_sourcefrq_lcnews']
    ril_rural = grouped1_riley_rural['Q13_NM_sourcefrq_lcnews']
    plat_urban = grouped1_platte_urban['Q13_NM_sourcefrq_lcnews']
    plat_suburban = grouped1_platte_suburban['Q13_NM_sourcefrq_lcnews']
    local_news_network_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural,
                               'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban, 'PS': plat_suburban,
                               'PU': plat_urban}
    return local_news_network_dict


def nodisaster_cable_news_nw():
    """
    This collects the frequency of interactions of each user with cable news when there is no disaster
    :return:
    """

    urban = grouped2_urban['Q13_NM_sourcefrq_news']
    suburban = grouped2_suburban['Q13_NM_sourcefrq_news']
    rural = grouped2_rural['Q13_NM_sourcefrq_news']

    buch_suburban = grouped1_buchanan_suburban['Q13_NM_sourcefrq_news']
    buch_rural = grouped1_buchanan_rural['Q13_NM_sourcefrq_news']
    ril_urban = grouped1_riley_urban['Q13_NM_sourcefrq_news']
    ril_suburban = grouped1_riley_suburban['Q13_NM_sourcefrq_news']
    ril_rural = grouped1_riley_rural['Q13_NM_sourcefrq_news']
    plat_urban = grouped1_platte_urban['Q13_NM_sourcefrq_news']
    plat_suburban = grouped1_platte_suburban['Q13_NM_sourcefrq_news']
    cable_news_network_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural,
                               'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban, 'PS': plat_suburban,
                               'PU': plat_urban}
    return cable_news_network_dict


def nodisaster_gov_nw():
    urban = grouped2_urban['Q13_NM_sourcefrq_gov']
    suburban = grouped2_suburban['Q13_NM_sourcefrq_gov']
    rural = grouped2_rural['Q13_NM_sourcefrq_gov']

    buch_suburban = grouped1_buchanan_suburban['Q13_NM_sourcefrq_gov']
    buch_rural = grouped1_buchanan_rural['Q13_NM_sourcefrq_gov']
    ril_urban = grouped1_riley_urban['Q13_NM_sourcefrq_gov']
    ril_suburban = grouped1_riley_suburban['Q13_NM_sourcefrq_gov']
    ril_rural = grouped1_riley_rural['Q13_NM_sourcefrq_gov']
    plat_urban = grouped1_platte_urban['Q13_NM_sourcefrq_gov']
    plat_suburban = grouped1_platte_suburban['Q13_NM_sourcefrq_gov']
    gov_network_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural, 'RR': ril_rural,
                        'RS': ril_suburban, 'RU': ril_urban, 'PS': plat_suburban, 'PU': plat_urban}
    return gov_network_dict


def nodisaster_print_nw():
    urban = grouped2_urban['Q13_NM_sourcefrq_print']
    suburban = grouped2_suburban['Q13_NM_sourcefrq_print']
    rural = grouped2_rural['Q13_NM_sourcefrq_print']

    buch_suburban = grouped1_buchanan_suburban['Q13_NM_sourcefrq_print']
    buch_rural = grouped1_buchanan_rural['Q13_NM_sourcefrq_print']
    ril_urban = grouped1_riley_urban['Q13_NM_sourcefrq_print']
    ril_suburban = grouped1_riley_suburban['Q13_NM_sourcefrq_print']
    ril_rural = grouped1_riley_rural['Q13_NM_sourcefrq_print']
    plat_urban = grouped1_platte_urban['Q13_NM_sourcefrq_print']
    plat_suburban = grouped1_platte_suburban['Q13_NM_sourcefrq_print']
    print_network_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural, 'RR': ril_rural,
                          'RS': ril_suburban, 'RU': ril_urban, 'PS': plat_suburban, 'PU': plat_urban}
    return print_network_dict


def nodisaster_social_nw():
    urban = grouped2_urban['Q13_NM_sourcefrq_sns']
    suburban = grouped2_suburban['Q13_NM_sourcefrq_sns']
    rural = grouped2_rural['Q13_NM_sourcefrq_sns']

    buch_suburban = grouped1_buchanan_suburban['Q13_NM_sourcefrq_sns']
    buch_rural = grouped1_buchanan_rural['Q13_NM_sourcefrq_sns']
    ril_urban = grouped1_riley_urban['Q13_NM_sourcefrq_sns']
    ril_suburban = grouped1_riley_suburban['Q13_NM_sourcefrq_sns']
    ril_rural = grouped1_riley_rural['Q13_NM_sourcefrq_sns']
    plat_urban = grouped1_platte_urban['Q13_NM_sourcefrq_sns']
    plat_suburban = grouped1_platte_suburban['Q13_NM_sourcefrq_sns']
    social_network_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural,
                           'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban, 'PS': plat_suburban, 'PU': plat_urban}
    return social_network_dict


# no disaster trust
def nodisaster_gov_nw_trust():
    urban = trust_filter(grouped2_urban['Q13_NM_trustsource_gov'])
    suburban = trust_filter(grouped2_suburban['Q13_NM_trustsource_gov'])
    rural = trust_filter(grouped2_rural['Q13_NM_trustsource_gov'])

    buch_suburban = trust_filter(grouped1_buchanan_suburban['Q13_NM_trustsource_gov'])
    buch_rural = trust_filter(grouped1_buchanan_rural['Q13_NM_trustsource_gov'])
    ril_urban = trust_filter(grouped1_riley_urban['Q13_NM_trustsource_gov'])
    ril_suburban = trust_filter(grouped1_riley_suburban['Q13_NM_trustsource_gov'])
    ril_rural = trust_filter(grouped1_riley_rural['Q13_NM_trustsource_gov'])
    plat_urban = trust_filter(grouped1_platte_urban['Q13_NM_trustsource_gov'])
    plat_suburban = trust_filter(grouped1_platte_suburban['Q13_NM_trustsource_gov'])

    gov_network_trust_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural,
                              'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                              'PS': plat_suburban, 'PU': plat_urban}
    return gov_network_trust_dict


def nodisaster_cable_news_nw_trust():
    urban = trust_filter(grouped2_urban['Q13_NM_trustsource_news'])
    suburban = trust_filter(grouped2_suburban['Q13_NM_trustsource_news'])
    rural = trust_filter(grouped2_rural['Q13_NM_trustsource_news'])

    buch_suburban = trust_filter(grouped1_buchanan_suburban['Q13_NM_trustsource_news'])
    buch_rural = trust_filter(grouped1_buchanan_rural['Q13_NM_trustsource_news'])
    ril_urban = trust_filter(grouped1_riley_urban['Q13_NM_trustsource_news'])
    ril_suburban = trust_filter(grouped1_riley_suburban['Q13_NM_trustsource_news'])
    ril_rural = trust_filter(grouped1_riley_rural['Q13_NM_trustsource_news'])
    plat_urban = trust_filter(grouped1_platte_urban['Q13_NM_trustsource_news'])
    plat_suburban = trust_filter(grouped1_platte_suburban['Q13_NM_trustsource_news'])
    news_network_trust_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural,
                               'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                               'PS': plat_suburban, 'PU': plat_urban}
    return news_network_trust_dict


def nodisaster_lcnews_nw_trust():
    urban = trust_filter(grouped2_urban['Q13_NM_trustsource_lcnews'])
    suburban = trust_filter(grouped2_suburban['Q13_NM_trustsource_lcnews'])
    rural = trust_filter(grouped2_rural['Q13_NM_trustsource_lcnews'])

    buch_suburban = trust_filter(grouped1_buchanan_suburban['Q13_NM_trustsource_lcnews'])
    buch_rural = trust_filter(grouped1_buchanan_rural['Q13_NM_trustsource_lcnews'])
    ril_urban = trust_filter(grouped1_riley_urban['Q13_NM_trustsource_lcnews'])
    ril_suburban = trust_filter(grouped1_riley_suburban['Q13_NM_trustsource_lcnews'])
    ril_rural = trust_filter(grouped1_riley_rural['Q13_NM_trustsource_lcnews'])
    plat_urban = trust_filter(grouped1_platte_urban['Q13_NM_trustsource_lcnews'])
    plat_suburban = trust_filter(grouped1_platte_suburban['Q13_NM_trustsource_lcnews'])
    lcnews_network_trust_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural,
                                 'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                                 'PS': plat_suburban, 'PU': plat_urban}
    return lcnews_network_trust_dict


def nodisaster_print_nw_trust():
    all_data = collect_users()
    urban, suburban, rural = group_data_urban_gradient(all_data)

    urban = trust_filter(urban['Q13_NM_trustsource_print'])
    suburban = trust_filter(suburban['Q13_NM_trustsource_print'])
    rural = trust_filter(rural['Q13_NM_trustsource_print'])

    buch_suburban = trust_filter(grouped1_buchanan_suburban['Q13_NM_trustsource_print'])
    buch_rural = trust_filter(grouped1_buchanan_rural['Q13_NM_trustsource_print'])
    ril_urban = trust_filter(grouped1_riley_urban['Q13_NM_trustsource_print'])
    ril_suburban = trust_filter(grouped1_riley_suburban['Q13_NM_trustsource_print'])
    ril_rural = trust_filter(grouped1_riley_rural['Q13_NM_trustsource_print'])
    plat_urban = trust_filter(grouped1_platte_urban['Q13_NM_trustsource_print'])
    plat_suburban = trust_filter(grouped1_platte_suburban['Q13_NM_trustsource_print'])
    print_network_trust_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural,
                                'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                                'PS': plat_suburban, 'PU': plat_urban}
    return print_network_trust_dict


def nodisaster_social_nw_trust():
    all_data = collect_users()
    urban, suburban, rural = group_data_urban_gradient(all_data)

    urban = trust_filter(urban['Q13_NM_trustsource_sns'])
    suburban = trust_filter(suburban['Q13_NM_trustsource_sns'])
    rural = trust_filter(rural['Q13_NM_trustsource_sns'])

    buch_suburban = trust_filter(grouped1_buchanan_suburban['Q13_NM_trustsource_sns'])
    buch_rural = trust_filter(grouped1_buchanan_rural['Q13_NM_trustsource_sns'])
    ril_urban = trust_filter(grouped1_riley_urban['Q13_NM_trustsource_sns'])
    ril_suburban = trust_filter(grouped1_riley_suburban['Q13_NM_trustsource_sns'])
    ril_rural = trust_filter(grouped1_riley_rural['Q13_NM_trustsource_sns'])
    plat_urban = trust_filter(grouped1_platte_urban['Q13_NM_trustsource_sns'])
    plat_suburban = trust_filter(grouped1_platte_suburban['Q13_NM_trustsource_sns'])
    social_network_trust_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural,
                                 'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                                 'PS': plat_suburban, 'PU': plat_urban}
    return social_network_trust_dict


##For Disaster
def disaster_lcnews_nw():
    urban = grouped2_urban['Q14_ND_sourcefrq_lcnews']
    suburban = grouped2_suburban['Q14_ND_sourcefrq_lcnews']
    rural = grouped2_rural['Q14_ND_sourcefrq_lcnews']
    buch_suburban = grouped1_buchanan_suburban['Q14_ND_sourcefrq_lcnews']
    buch_rural = grouped1_buchanan_rural['Q14_ND_sourcefrq_lcnews']
    ril_urban = grouped1_riley_urban['Q14_ND_sourcefrq_lcnews']
    ril_suburban = grouped1_riley_suburban['Q14_ND_sourcefrq_lcnews']
    ril_rural = grouped1_riley_rural['Q14_ND_sourcefrq_lcnews']
    plat_urban = grouped1_platte_urban['Q14_ND_sourcefrq_lcnews']
    plat_suburban = grouped1_platte_suburban['Q14_ND_sourcefrq_lcnews']
    lcnews_network_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural,
                           'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban, 'PS': plat_suburban, 'PU': plat_urban}
    return lcnews_network_dict


def disaster_cable_news_nw():
    urban = grouped2_urban['Q14_ND_sourcefrq_news']
    suburban = grouped2_suburban['Q14_ND_sourcefrq_news']
    rural = grouped2_rural['Q14_ND_sourcefrq_news']
    buch_suburban = grouped1_buchanan_suburban['Q14_ND_sourcefrq_news']
    buch_rural = grouped1_buchanan_rural['Q14_ND_sourcefrq_news']
    ril_urban = grouped1_riley_urban['Q14_ND_sourcefrq_news']
    ril_suburban = grouped1_riley_suburban['Q14_ND_sourcefrq_news']
    ril_rural = grouped1_riley_rural['Q14_ND_sourcefrq_news']
    plat_urban = grouped1_platte_urban['Q14_ND_sourcefrq_news']
    plat_suburban = grouped1_platte_suburban['Q14_ND_sourcefrq_news']
    news_network_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural, 'RR': ril_rural,
                         'RS': ril_suburban, 'RU': ril_urban, 'PS': plat_suburban, 'PU': plat_urban}
    return news_network_dict


def disaster_gov_nw():
    urban = grouped2_urban['Q14_ND_sourcefrq_gov']
    suburban = grouped2_suburban['Q14_ND_sourcefrq_gov']
    rural = grouped2_rural['Q14_ND_sourcefrq_gov']
    buch_suburban = grouped1_buchanan_suburban['Q14_ND_sourcefrq_gov']
    buch_rural = grouped1_buchanan_rural['Q14_ND_sourcefrq_gov']
    ril_urban = grouped1_riley_urban['Q14_ND_sourcefrq_gov']
    ril_suburban = grouped1_riley_suburban['Q14_ND_sourcefrq_gov']
    ril_rural = grouped1_riley_rural['Q14_ND_sourcefrq_gov']
    plat_urban = grouped1_platte_urban['Q14_ND_sourcefrq_gov']
    plat_suburban = grouped1_platte_suburban['Q14_ND_sourcefrq_gov']
    gov_network_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural, 'RR': ril_rural,
                        'RS': ril_suburban, 'RU': ril_urban, 'PS': plat_suburban, 'PU': plat_urban}
    return gov_network_dict


def disaster_print_nw():
    urban = grouped2_urban['Q14_ND_sourcefrq_print']
    suburban = grouped2_suburban['Q14_ND_sourcefrq_print']
    rural = grouped2_rural['Q14_ND_sourcefrq_print']
    buch_suburban = grouped1_buchanan_suburban['Q14_ND_sourcefrq_print']
    buch_rural = grouped1_buchanan_rural['Q14_ND_sourcefrq_print']
    ril_urban = grouped1_riley_urban['Q14_ND_sourcefrq_print']
    ril_suburban = grouped1_riley_suburban['Q14_ND_sourcefrq_print']
    ril_rural = grouped1_riley_rural['Q14_ND_sourcefrq_print']
    plat_urban = grouped1_platte_urban['Q14_ND_sourcefrq_print']
    plat_suburban = grouped1_platte_suburban['Q14_ND_sourcefrq_print']
    print_network_dict = {'R': rural, 'S': suburban, 'U': urban, 'BS': buch_suburban, 'BR': buch_rural, 'RR': ril_rural,
                          'RS': ril_suburban, 'RU': ril_urban, 'PS': plat_suburban, 'PU': plat_urban}
    return print_network_dict


def disaster_social_nw():
    urban = grouped2_urban['Q14_ND_sourcefrq_sns']
    suburban = grouped2_suburban['Q14_ND_sourcefrq_sns']
    rural = grouped2_rural['Q14_ND_sourcefrq_sns']
    buch_suburban = grouped1_buchanan_suburban['Q14_ND_sourcefrq_sns']
    buch_rural = grouped1_buchanan_rural['Q14_ND_sourcefrq_sns']
    ril_urban = grouped1_riley_urban['Q14_ND_sourcefrq_sns']
    ril_suburban = grouped1_riley_suburban['Q14_ND_sourcefrq_sns']
    ril_rural = grouped1_riley_rural['Q14_ND_sourcefrq_sns']
    plat_urban = grouped1_platte_urban['Q14_ND_sourcefrq_sns']
    plat_suburban = grouped1_platte_suburban['Q14_ND_sourcefrq_sns']
    social_network_dict = {'R': rural, 'S': suburban, 'U': urban, 'BR': buch_rural,
                           'BS': buch_suburban, 'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                           'PS': plat_suburban, 'PU': plat_urban}
    return social_network_dict


# Trust for sources of news during disaster
def disaster_lcnews_nw_trust():
    urban = trust_filter(grouped2_urban['Q14_ND_trustsource_lcnews'])
    suburban = trust_filter(grouped2_suburban['Q14_ND_trustsource_lcnews'])
    rural = trust_filter(grouped2_rural['Q14_ND_trustsource_lcnews'])
    buch_suburban = trust_filter(grouped1_buchanan_suburban['Q14_ND_trustsource_lcnews'])
    buch_rural = trust_filter(grouped1_buchanan_rural['Q14_ND_trustsource_lcnews'])
    ril_urban = trust_filter(grouped1_riley_urban['Q14_ND_trustsource_lcnews'])
    ril_suburban = trust_filter(grouped1_riley_suburban['Q14_ND_trustsource_lcnews'])
    ril_rural = trust_filter(grouped1_riley_rural['Q14_ND_trustsource_lcnews'])
    plat_urban = trust_filter(grouped1_platte_urban['Q14_ND_trustsource_lcnews'])
    plat_suburban = trust_filter(grouped1_platte_suburban['Q14_ND_trustsource_lcnews'])
    lcnews_network_trust_dict = {'R': rural, 'S': suburban, 'U': urban, 'BR': buch_rural,
                                 'BS': buch_suburban, 'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                                 'PS': plat_suburban, 'PU': plat_urban}
    return lcnews_network_trust_dict


def disaster_cable_news_nw_trust():
    urban = trust_filter(grouped2_urban['Q14_ND_trustsource_news'])
    suburban = trust_filter(grouped2_suburban['Q14_ND_trustsource_news'])
    rural = trust_filter(grouped2_rural['Q14_ND_trustsource_news'])
    buch_suburban = trust_filter(grouped1_buchanan_suburban['Q14_ND_trustsource_news'])
    buch_rural = trust_filter(grouped1_buchanan_rural['Q14_ND_trustsource_news'])
    ril_urban = trust_filter(grouped1_riley_urban['Q14_ND_trustsource_news'])
    ril_suburban = trust_filter(grouped1_riley_suburban['Q14_ND_trustsource_news'])
    ril_rural = trust_filter(grouped1_riley_rural['Q14_ND_trustsource_news'])
    plat_urban = trust_filter(grouped1_platte_urban['Q14_ND_trustsource_news'])
    plat_suburban = trust_filter(grouped1_platte_suburban['Q14_ND_trustsource_news'])
    news_network_trust_dict = {'R': rural, 'S': suburban, 'U': urban, 'BR': buch_rural,
                               'BS': buch_suburban, 'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                               'PS': plat_suburban, 'PU': plat_urban}
    return news_network_trust_dict


def disaster_gov_nw_trust():
    urban = trust_filter(grouped2_urban['Q14_ND_trustsource_gov'])
    suburban = trust_filter(grouped2_suburban['Q14_ND_trustsource_gov'])
    rural = trust_filter(grouped2_rural['Q14_ND_trustsource_gov'])
    buch_suburban = trust_filter(grouped1_buchanan_suburban['Q14_ND_trustsource_gov'])
    buch_rural = trust_filter(grouped1_buchanan_rural['Q14_ND_trustsource_gov'])
    ril_urban = trust_filter(grouped1_riley_urban['Q14_ND_trustsource_gov'])
    ril_suburban = trust_filter(grouped1_riley_suburban['Q14_ND_trustsource_gov'])
    ril_rural = trust_filter(grouped1_riley_rural['Q14_ND_trustsource_gov'])
    plat_urban = trust_filter(grouped1_platte_urban['Q14_ND_trustsource_gov'])
    plat_suburban = trust_filter(grouped1_platte_suburban['Q14_ND_trustsource_gov'])
    gov_network_trust_dict = {'R': rural, 'S': suburban, 'U': urban, 'BR': buch_rural,
                              'BS': buch_suburban, 'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                              'PS': plat_suburban, 'PU': plat_urban}
    return gov_network_trust_dict


def disaster_print_nw_trust():
    urban = trust_filter(grouped2_urban['Q14_ND_trustsource_print'])
    suburban = trust_filter(grouped2_suburban['Q14_ND_trustsource_print'])
    rural = trust_filter(grouped2_rural['Q14_ND_trustsource_print'])
    buch_suburban = trust_filter(grouped1_buchanan_suburban['Q14_ND_trustsource_print'])
    buch_rural = trust_filter(grouped1_buchanan_rural['Q14_ND_trustsource_print'])
    ril_urban = trust_filter(grouped1_riley_urban['Q14_ND_trustsource_print'])
    ril_suburban = trust_filter(grouped1_riley_suburban['Q14_ND_trustsource_print'])
    ril_rural = trust_filter(grouped1_riley_rural['Q14_ND_trustsource_print'])
    plat_urban = trust_filter(grouped1_platte_urban['Q14_ND_trustsource_print'])
    plat_suburban = trust_filter(grouped1_platte_suburban['Q14_ND_trustsource_print'])
    print_network_trust_dict = {'R': rural, 'S': suburban, 'U': urban, 'BR': buch_rural,
                                'BS': buch_suburban, 'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                                'PS': plat_suburban, 'PU': plat_urban}
    return print_network_trust_dict


def disaster_social_nw_trust():
    urban = trust_filter(grouped2_urban['Q14_ND_trustsource_sns'])
    suburban = trust_filter(grouped2_suburban['Q14_ND_trustsource_sns'])
    rural = trust_filter(grouped2_rural['Q14_ND_trustsource_sns'])
    buch_suburban = trust_filter(grouped1_buchanan_suburban['Q14_ND_trustsource_sns'])
    buch_rural = trust_filter(grouped1_buchanan_rural['Q14_ND_trustsource_sns'])
    ril_urban = trust_filter(grouped1_riley_urban['Q14_ND_trustsource_sns'])
    ril_suburban = trust_filter(grouped1_riley_suburban['Q14_ND_trustsource_sns'])
    ril_rural = trust_filter(grouped1_riley_rural['Q14_ND_trustsource_sns'])
    plat_urban = trust_filter(grouped1_platte_urban['Q14_ND_trustsource_sns'])
    plat_suburban = trust_filter(grouped1_platte_suburban['Q14_ND_trustsource_sns'])
    social_network_trust_dict = {'R': rural, 'S': suburban, 'U': urban, 'BR': buch_rural,
                                 'BS': buch_suburban, 'RR': ril_rural, 'RS': ril_suburban, 'RU': ril_urban,
                                 'PS': plat_suburban, 'PU': plat_urban}
    return social_network_trust_dict


def attributes_rural():
    """
    When there is no disaster, this returns the degree distribution and weekly number of interactions among people living in rural communities in this county
    :return:
    """
    distb = grouped2_rural['Q4_ITR_people']
    weekly_intera_rural = grouped2_rural[
        ['Q11_NM_interact_P1', 'Q11_NM_interact_P2', 'Q11_NM_interact_P3', 'Q11_NM_interact_P4',
         'Q11_NM_interact_P5']].replace(-99, np.nan)
    cleaned_weekly_intera_rural = weekly_intera_rural.mean(axis=1).round()
    cleaned_weekly_intera_rural = cleaned_weekly_intera_rural[~np.isnan(cleaned_weekly_intera_rural)]
    return distb, cleaned_weekly_intera_rural


def attributes_suburban():
    distb = grouped2_suburban['Q4_ITR_people']
    weekly_intera_suburban = grouped2_rural[
        ['Q11_NM_interact_P1', 'Q11_NM_interact_P2', 'Q11_NM_interact_P3', 'Q11_NM_interact_P4',
         'Q11_NM_interact_P5']].replace(-99, np.nan)
    cleaned_weekly_intera_suburban = weekly_intera_suburban.mean(axis=1).round()
    cleaned_weekly_intera_suburban = cleaned_weekly_intera_suburban[~np.isnan(cleaned_weekly_intera_suburban)]
    return distb, cleaned_weekly_intera_suburban


def attributes_urban():
    distb = grouped2_urban['Q4_ITR_people']
    weekly_intera_urban = grouped2_urban[
        ['Q11_NM_interact_P1', 'Q11_NM_interact_P2', 'Q11_NM_interact_P3', 'Q11_NM_interact_P4',
         'Q11_NM_interact_P5']].replace(-99, np.nan)
    cleaned_weekly_intera_urban = weekly_intera_urban.mean(axis=1).round()
    cleaned_weekly_intera_urban = cleaned_weekly_intera_urban[~np.isnan(cleaned_weekly_intera_urban)]
    return distb, cleaned_weekly_intera_urban


def attributes_buch_rural():
    distb = grouped1_buchanan_rural['Q4_ITR_people']
    weekly_intera_buch_rur = grouped1_buchanan_rural[
        ['Q11_NM_interact_P1', 'Q11_NM_interact_P2', 'Q11_NM_interact_P3', 'Q11_NM_interact_P4',
         'Q11_NM_interact_P5']].replace(-99, np.nan)
    cleaned_weekly_intera_buch_rur = weekly_intera_buch_rur.mean(axis=1).round()
    cleaned_weekly_intera_buch_rur = cleaned_weekly_intera_buch_rur[~np.isnan(cleaned_weekly_intera_buch_rur)]
    return distb, cleaned_weekly_intera_buch_rur


def attributes_buch_suburban():
    distb = grouped1_buchanan_suburban['Q4_ITR_people']
    weekly_intera_buch_suburban = grouped1_buchanan_suburban[
        ['Q11_NM_interact_P1', 'Q11_NM_interact_P2', 'Q11_NM_interact_P3', 'Q11_NM_interact_P4',
         'Q11_NM_interact_P5']].replace(-99, np.nan)
    cleaned_weekly_intera_buch_suburban = weekly_intera_buch_suburban.mean(axis=1).round()
    cleaned_weekly_intera_buch_suburban = cleaned_weekly_intera_buch_suburban[
        ~np.isnan(cleaned_weekly_intera_buch_suburban)]
    return distb, cleaned_weekly_intera_buch_suburban


def attributes_ril_rural():
    distb = grouped1_riley_rural['Q4_ITR_people']
    weekly_intera_ril_rural = grouped1_riley_rural[
        ['Q11_NM_interact_P1', 'Q11_NM_interact_P2', 'Q11_NM_interact_P3', 'Q11_NM_interact_P4',
         'Q11_NM_interact_P5']].replace(-99, np.nan)
    cleaned_weekly_intera_ril_rural = weekly_intera_ril_rural.mean(axis=1).round()
    cleaned_weekly_intera_ril_rural = cleaned_weekly_intera_ril_rural[~np.isnan(cleaned_weekly_intera_ril_rural)]
    return distb, cleaned_weekly_intera_ril_rural


def attributes_ril_suburban():
    distb = grouped1_riley_suburban['Q4_ITR_people']
    weekly_intera_ril_suburban = grouped1_riley_suburban[
        ['Q11_NM_interact_P1', 'Q11_NM_interact_P2', 'Q11_NM_interact_P3', 'Q11_NM_interact_P4',
         'Q11_NM_interact_P5']].replace(-99, np.nan)
    cleaned_weekly_intera_ril_suburban = weekly_intera_ril_suburban.mean(axis=1).round()
    cleaned_weekly_intera_ril_suburban = cleaned_weekly_intera_ril_suburban[
        ~np.isnan(cleaned_weekly_intera_ril_suburban)]
    return distb, cleaned_weekly_intera_ril_suburban


def attributes_ril_urban():
    distb = grouped1_riley_urban['Q4_ITR_people']
    weekly_intera_ril_urban = grouped1_riley_urban[
        ['Q11_NM_interact_P1', 'Q11_NM_interact_P2', 'Q11_NM_interact_P3', 'Q11_NM_interact_P4',
         'Q11_NM_interact_P5']].replace(-99, np.nan)
    cleaned_weekly_intera_ril_urban = weekly_intera_ril_urban.mean(axis=1).round()
    cleaned_weekly_intera_ril_urban = cleaned_weekly_intera_ril_urban[~np.isnan(cleaned_weekly_intera_ril_urban)]
    return distb, cleaned_weekly_intera_ril_urban


def attributes_pla_suburban():
    distb = grouped1_platte_suburban['Q4_ITR_people']
    weekly_intera_pla_suburban = grouped1_platte_suburban[
        ['Q11_NM_interact_P1', 'Q11_NM_interact_P2', 'Q11_NM_interact_P3', 'Q11_NM_interact_P4',
         'Q11_NM_interact_P5']].replace(-99, np.nan)
    cleaned_weekly_intera_pla_suburban = weekly_intera_pla_suburban.mean(axis=1).round()
    cleaned_weekly_intera_pla_suburban = cleaned_weekly_intera_pla_suburban[
        ~np.isnan(cleaned_weekly_intera_pla_suburban)]
    return distb, cleaned_weekly_intera_pla_suburban


def attributes_pla_urban():
    distb = grouped1_platte_urban['Q4_ITR_people']
    weekly_intera_pla_urban = grouped1_platte_urban[
        ['Q11_NM_interact_P1', 'Q11_NM_interact_P2', 'Q11_NM_interact_P3', 'Q11_NM_interact_P4',
         'Q11_NM_interact_P5']].replace(-99, np.nan)
    cleaned_weekly_intera_pla_urban = weekly_intera_pla_urban.mean(axis=1).round()
    cleaned_weekly_intera_pla_urban = cleaned_weekly_intera_pla_urban[~np.isnan(cleaned_weekly_intera_pla_urban)]
    return distb, cleaned_weekly_intera_pla_urban


###TRUST
# Learn interaction trust by district and by area

def attributes_rural_trust():
    """
    Collects trust for 5 most trusted individuals a user interacts with when there is no disaster
    :return:
    """
    trust_rural = grouped2_rural[
        ['Q11_NM_trust_P1', 'Q11_NM_trust_P2', 'Q11_NM_trust_P3', 'Q11_NM_trust_P4', 'Q11_NM_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_rural = trust_rural.mean(axis=1).round()
    trust_rural = trust_rural[~np.isnan(trust_rural)]
    trust_rural = trust_filter(trust_rural)
    return trust_rural


def attributes_suburban_trust():
    trust_suburban = grouped2_suburban[
        ['Q11_NM_trust_P1', 'Q11_NM_trust_P2', 'Q11_NM_trust_P3', 'Q11_NM_trust_P4', 'Q11_NM_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_suburban = trust_suburban.mean(axis=1).round()
    trust_suburban = trust_suburban[~np.isnan(trust_suburban)]
    trust_suburban = trust_filter(trust_suburban)
    return trust_suburban


def attributes_urban_trust():
    trust_urban = grouped2_urban[
        ['Q11_NM_trust_P1', 'Q11_NM_trust_P2', 'Q11_NM_trust_P3', 'Q11_NM_trust_P4', 'Q11_NM_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_urban = trust_urban.mean(axis=1).round()
    trust_urban = trust_urban[~np.isnan(trust_urban)]
    trust_urban = trust_filter(trust_urban)
    return trust_urban


def attributes_buch_rural_trust():
    trust_buch_rur = grouped1_buchanan_rural[
        ['Q11_NM_trust_P1', 'Q11_NM_trust_P2', 'Q11_NM_trust_P3', 'Q11_NM_trust_P4', 'Q11_NM_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_buch_rur = trust_buch_rur.mean(axis=1).round()
    trust_buch_rur = trust_buch_rur[~np.isnan(trust_buch_rur)]
    trust_buch_rur = trust_filter(trust_buch_rur)
    return trust_buch_rur


def attributes_buch_suburban_trust():
    trust_buch_sub = grouped1_buchanan_suburban[
        ['Q11_NM_trust_P1', 'Q11_NM_trust_P2', 'Q11_NM_trust_P3', 'Q11_NM_trust_P4', 'Q11_NM_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_buch_sub = trust_buch_sub.mean(axis=1).round()
    trust_buch_sub = trust_buch_sub[~np.isnan(trust_buch_sub)]
    trust_buch_sub = trust_filter(trust_buch_sub)
    return trust_buch_sub


def attributes_ril_rural_trust():
    trust_ril_rur = grouped1_riley_rural[
        ['Q11_NM_trust_P1', 'Q11_NM_trust_P2', 'Q11_NM_trust_P3', 'Q11_NM_trust_P4', 'Q11_NM_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_ril_rur = trust_ril_rur.mean(axis=1).round()
    trust_ril_rur = trust_ril_rur[~np.isnan(trust_ril_rur)]
    trust_ril_rur = trust_filter(trust_ril_rur)
    return trust_ril_rur


def attributes_ril_suburban_trust():
    trust_ril_sub = grouped1_riley_suburban[
        ['Q11_NM_trust_P1', 'Q11_NM_trust_P2', 'Q11_NM_trust_P3', 'Q11_NM_trust_P4', 'Q11_NM_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_ril_sub = trust_ril_sub.mean(axis=1).round()
    trust_ril_sub = trust_ril_sub[~np.isnan(trust_ril_sub)]
    trust_ril_sub = trust_filter(trust_ril_sub)
    return trust_ril_sub


def attributes_ril_urban_trust():
    trust_ril_urb = grouped1_riley_urban[
        ['Q11_NM_trust_P1', 'Q11_NM_trust_P2', 'Q11_NM_trust_P3', 'Q11_NM_trust_P4', 'Q11_NM_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_ril_urb = trust_ril_urb.mean(axis=1).round()
    trust_ril_urb = trust_ril_urb[~np.isnan(trust_ril_urb)]
    trust_ril_urb = trust_filter(trust_ril_urb)
    return trust_ril_urb


def attributes_pla_suburban_trust():
    trust_plat_sub = grouped1_platte_suburban[
        ['Q11_NM_trust_P1', 'Q11_NM_trust_P2', 'Q11_NM_trust_P3', 'Q11_NM_trust_P4', 'Q11_NM_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_plat_sub = trust_plat_sub.mean(axis=1).round()
    trust_plat_sub = trust_plat_sub[~np.isnan(trust_plat_sub)]
    trust_plat_sub = trust_filter(trust_plat_sub)
    return trust_plat_sub


def attributes_pla_urban_trust():
    trust_plat_urb = grouped1_platte_urban[
        ['Q11_NM_trust_P1', 'Q11_NM_trust_P2', 'Q11_NM_trust_P3', 'Q11_NM_trust_P4', 'Q11_NM_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_plat_urb = trust_plat_urb.mean(axis=1).round()
    trust_plat_urb = trust_plat_urb[~np.isnan(trust_plat_urb)]
    trust_plat_urb = trust_filter(trust_plat_urb)
    return trust_plat_urb


# Disaster related data
def attributes_rural_dis():
    """
    This returns the distribution of the rural communities from data and the corresponding weekly interactions
    :return:
    """
    all_data = collect_users()
    _, _, rural = group_data_urban_gradient(all_data)
    distb = rural['Q4_ITR_people']
    weekly_intera_rural = rural[['Q12_ND_interact_1', 'Q12_ND_interact_2', 'Q12_ND_interact_3', 'Q12_ND_interact_4',
                                 'Q12_ND_interact_5']].replace(-99, np.nan)
    cleaned_weekly_intera_rural = weekly_intera_rural.mean(axis=1).round()
    cleaned_weekly_intera_rural = cleaned_weekly_intera_rural[~np.isnan(cleaned_weekly_intera_rural)]
    return distb, cleaned_weekly_intera_rural


def attributes_suburban_dis():
    all_data = collect_users()
    _, suburban, _ = group_data_urban_gradient(all_data)
    distb = suburban['Q4_ITR_people']
    weekly_intera_suburban = suburban[
        ['Q12_ND_interact_1', 'Q12_ND_interact_2', 'Q12_ND_interact_3', 'Q12_ND_interact_4',
         'Q12_ND_interact_5']].replace(-99, np.nan)
    cleaned_weekly_intera_suburban = weekly_intera_suburban.mean(axis=1).round()
    cleaned_weekly_intera_suburban = cleaned_weekly_intera_suburban[~np.isnan(cleaned_weekly_intera_suburban)]
    return distb, cleaned_weekly_intera_suburban


def attributes_urban_dis():
    all_data = collect_users()
    urban, _, _ = group_data_urban_gradient(all_data)
    distb = urban['Q4_ITR_people']
    weekly_intera_urban = urban[['Q12_ND_interact_1', 'Q12_ND_interact_2', 'Q12_ND_interact_3', 'Q12_ND_interact_4',
                                 'Q12_ND_interact_5']].replace(-99, np.nan)
    cleaned_weekly_intera_urban = weekly_intera_urban.mean(axis=1).round()
    cleaned_weekly_intera_urban = cleaned_weekly_intera_urban[~np.isnan(cleaned_weekly_intera_urban)]
    return distb, cleaned_weekly_intera_urban


def attributes_buch_rural_dis():
    all_data = collect_users()
    _, _, buch_rural, _, _, _, _, _, _ = group_data_district(all_data)
    distb = buch_rural['Q4_ITR_people']
    weekly_intera_buch_rur = buch_rural[
        ['Q12_ND_interact_1', 'Q12_ND_interact_2', 'Q12_ND_interact_3', 'Q12_ND_interact_4',
         'Q12_ND_interact_5']].replace(-99, np.nan)
    cleaned_weekly_intera_buch_rur = weekly_intera_buch_rur.mean(axis=1).round()
    cleaned_weekly_intera_buch_rur = cleaned_weekly_intera_buch_rur[~np.isnan(cleaned_weekly_intera_buch_rur)]
    return distb, cleaned_weekly_intera_buch_rur


def attributes_buch_suburban_dis():
    all_data = collect_users()
    _, buch_suburban, _, _, _, _, _, _, _ = group_data_district(all_data)
    distb = buch_suburban['Q4_ITR_people']
    weekly_intera_buch_suburban = buch_suburban[
        ['Q12_ND_interact_1', 'Q12_ND_interact_2', 'Q12_ND_interact_3', 'Q12_ND_interact_4',
         'Q12_ND_interact_5']].replace(-99, np.nan)
    cleaned_weekly_intera_buch_suburban = weekly_intera_buch_suburban.mean(axis=1).round()
    cleaned_weekly_intera_buch_suburban = cleaned_weekly_intera_buch_suburban[
        ~np.isnan(cleaned_weekly_intera_buch_suburban)]
    return distb, cleaned_weekly_intera_buch_suburban


def attributes_ril_rural_dis():
    all_data = collect_users()
    _, _, _, _, _, ril_rural, _, _, _ = group_data_district(all_data)
    distb = ril_rural['Q4_ITR_people']
    weekly_intera_ril_rural = ril_rural[
        ['Q12_ND_interact_1', 'Q12_ND_interact_2', 'Q12_ND_interact_3', 'Q12_ND_interact_4',
         'Q12_ND_interact_5']].replace(-99, np.nan)
    cleaned_weekly_intera_ril_rural = weekly_intera_ril_rural.mean(axis=1).round()
    cleaned_weekly_intera_ril_rural = cleaned_weekly_intera_ril_rural[~np.isnan(cleaned_weekly_intera_ril_rural)]
    return distb, cleaned_weekly_intera_ril_rural


def attributes_ril_suburban_dis():
    all_data = collect_users()
    _, _, _, _, ril_suburban, _, _, _, _ = group_data_district(all_data)
    distb = ril_suburban['Q4_ITR_people']
    weekly_intera_ril_suburban = ril_suburban[
        ['Q12_ND_interact_1', 'Q12_ND_interact_2', 'Q12_ND_interact_3', 'Q12_ND_interact_4',
         'Q12_ND_interact_5']].replace(-99, np.nan)
    cleaned_weekly_intera_ril_suburban = weekly_intera_ril_suburban.mean(axis=1).round()
    cleaned_weekly_intera_ril_suburban = cleaned_weekly_intera_ril_suburban[
        ~np.isnan(cleaned_weekly_intera_ril_suburban)]
    return distb, cleaned_weekly_intera_ril_suburban


def attributes_ril_urban_dis():
    all_data = collect_users()
    _, _, _, ril_urban, _, _, _, _, _ = group_data_district(all_data)
    distb = ril_urban['Q4_ITR_people']
    weekly_intera_ril_urban = ril_urban[
        ['Q12_ND_interact_1', 'Q12_ND_interact_2', 'Q12_ND_interact_3', 'Q12_ND_interact_4',
         'Q12_ND_interact_5']].replace(-99, np.nan)
    cleaned_weekly_intera_ril_urban = weekly_intera_ril_urban.mean(axis=1).round()
    cleaned_weekly_intera_ril_urban = cleaned_weekly_intera_ril_urban[~np.isnan(cleaned_weekly_intera_ril_urban)]
    return distb, cleaned_weekly_intera_ril_urban


def attributes_pla_suburban_dis():
    all_data = collect_users()
    _, _, _, _, _, _, _, pla_suburban, _ = group_data_district(all_data)
    distb = pla_suburban['Q4_ITR_people']
    weekly_intera_pla_suburban = pla_suburban[
        ['Q12_ND_interact_1', 'Q12_ND_interact_2', 'Q12_ND_interact_3', 'Q12_ND_interact_4',
         'Q12_ND_interact_5']].replace(-99, np.nan)
    cleaned_weekly_intera_pla_suburban = weekly_intera_pla_suburban.mean(axis=1).round()
    cleaned_weekly_intera_pla_suburban = cleaned_weekly_intera_pla_suburban[
        ~np.isnan(cleaned_weekly_intera_pla_suburban)]
    return distb, cleaned_weekly_intera_pla_suburban


def attributes_pla_urban_dis():
    all_data = collect_users()
    _, _, _, _, _, _, pla_urban, _, _ = group_data_district(all_data)
    distb = pla_urban['Q4_ITR_people']
    weekly_intera_buch_rur = pla_urban[
        ['Q12_ND_interact_1', 'Q12_ND_interact_2', 'Q12_ND_interact_3', 'Q12_ND_interact_4',
         'Q12_ND_interact_5']].replace(-99, np.nan)
    cleaned_weekly_intera_pla_urban = weekly_intera_buch_rur.mean(axis=1).round()
    cleaned_weekly_intera_pla_urban = cleaned_weekly_intera_pla_urban[~np.isnan(cleaned_weekly_intera_pla_urban)]
    return distb, cleaned_weekly_intera_pla_urban


# For Disaster Trust
def attributes_rural_trust_dis():
    trust_rural = grouped2_rural[
        ['Q12_ND_trust_P1', 'Q12_ND_trust_P2', 'Q12_ND_trust_P3', 'Q12_ND_trust_P4', 'Q12_ND_trust_P5']].replace(-99,
                                                                                                                 np.nan)
    trust_rural = trust_rural.mean(axis=1).round()
    trust_rural = trust_rural[~np.isnan(trust_rural)]
    trust_rural = trust_filter(trust_rural)
    return trust_rural


def attributes_suburban_trust_dis():
    trust_suburban = grouped2_suburban[
        ['Q12_ND_trust_P1', 'Q12_ND_trust_P2', 'Q12_ND_trust_P3', 'Q12_ND_trust_P4', 'Q12_ND_trust_P5']]
    trust_suburban = trust_suburban.mean(axis=1).round()
    trust_suburban = trust_suburban[~np.isnan(trust_suburban)]
    trust_suburban = trust_filter(trust_suburban)
    return trust_suburban


def attributes_urban_trust_dis():
    trust_urban = grouped2_urban[
        ['Q12_ND_trust_P1', 'Q12_ND_trust_P2', 'Q12_ND_trust_P3', 'Q12_ND_trust_P4', 'Q12_ND_trust_P5']]
    trust_urban = trust_urban.mean(axis=1).round()
    trust_urban = trust_urban[~np.isnan(trust_urban)]
    trust_urban = trust_filter(trust_urban)
    return trust_urban


def attributes_buch_rural_trust_dis():
    trust_buch_rur = grouped1_buchanan_rural[
        ['Q12_ND_trust_P1', 'Q12_ND_trust_P2', 'Q12_ND_trust_P3', 'Q12_ND_trust_P4', 'Q12_ND_trust_P5']]
    trust_buch_rur = trust_buch_rur.mean(axis=1).round()
    trust_buch_rur = trust_buch_rur[~np.isnan(trust_buch_rur)]
    trust_buch_rur = trust_filter(trust_buch_rur)
    return trust_buch_rur


def attributes_buch_suburban_trust_dis():
    trust_buch_sub = grouped1_buchanan_suburban[
        ['Q12_ND_trust_P1', 'Q12_ND_trust_P2', 'Q12_ND_trust_P3', 'Q12_ND_trust_P4', 'Q12_ND_trust_P5']]
    trust_buch_sub = trust_buch_sub.mean(axis=1).round()
    trust_buch_sub = trust_buch_sub[~np.isnan(trust_buch_sub)]
    trust_buch_sub = trust_filter(trust_buch_sub)
    return trust_buch_sub


def attributes_ril_rural_trust_dis():
    trust_ril_rur = grouped1_riley_rural[
        ['Q12_ND_trust_P1', 'Q12_ND_trust_P2', 'Q12_ND_trust_P3', 'Q12_ND_trust_P4', 'Q12_ND_trust_P5']]
    trust_ril_rur = trust_ril_rur.mean(axis=1).round()
    trust_ril_rur = trust_ril_rur[~np.isnan(trust_ril_rur)]
    trust_ril_rur = trust_filter(trust_ril_rur)
    return trust_ril_rur


def attributes_ril_suburban_trust_dis():
    trust_ril_sub = grouped1_riley_suburban[
        ['Q12_ND_trust_P1', 'Q12_ND_trust_P2', 'Q12_ND_trust_P3', 'Q12_ND_trust_P4', 'Q12_ND_trust_P5']]
    trust_ril_sub = trust_ril_sub.mean(axis=1).round()
    trust_ril_sub = trust_ril_sub[~np.isnan(trust_ril_sub)]
    trust_ril_sub = trust_filter(trust_ril_sub)
    return trust_ril_sub


def attributes_ril_urban_trust_dis():
    trust_ril_urb = grouped1_riley_urban[
        ['Q12_ND_trust_P1', 'Q12_ND_trust_P2', 'Q12_ND_trust_P3', 'Q12_ND_trust_P4', 'Q12_ND_trust_P5']]
    trust_ril_urb = trust_ril_urb.mean(axis=1).round()
    trust_ril_urb = trust_ril_urb[~np.isnan(trust_ril_urb)]
    trust_ril_urb = trust_filter(trust_ril_urb)
    return trust_ril_urb


def attributes_pla_suburban_trust_dis():
    trust_plat_sub = grouped1_platte_suburban[
        ['Q12_ND_trust_P1', 'Q12_ND_trust_P2', 'Q12_ND_trust_P3', 'Q12_ND_trust_P4', 'Q12_ND_trust_P5']]
    trust_plat_sub = trust_plat_sub.mean(axis=1).round()
    trust_plat_sub = trust_plat_sub[~np.isnan(trust_plat_sub)]
    trust_plat_sub = trust_filter(trust_plat_sub)
    return trust_plat_sub


def attributes_pla_urban_trust_dis():
    trust_plat_urb = grouped1_platte_urban[
        ['Q12_ND_trust_P1', 'Q12_ND_trust_P2', 'Q12_ND_trust_P3', 'Q12_ND_trust_P4', 'Q12_ND_trust_P5']]
    trust_plat_urb = trust_plat_urb.mean(axis=1).round()
    trust_plat_urb = trust_plat_urb[~np.isnan(trust_plat_urb)]
    trust_plat_urb = trust_filter(trust_plat_urb)
    return trust_plat_urb


def source_interact_none():
    """
    :return: distribution of the contacts and frequency of interaction
    """
    source_interact_dict = {'R': attributes_rural(), 'S': attributes_suburban(), 'U': attributes_urban(),
                            'BR': attributes_buch_rural(), 'BS': attributes_buch_suburban(),
                            'RR': attributes_ril_rural(),
                            'RS': attributes_ril_suburban(), 'RU': attributes_ril_urban(),
                            'PS': attributes_pla_suburban(),
                            'PU': attributes_pla_urban()}
    return source_interact_dict


def trust_interact_none():
    """
    :return: trust that each survey respondant have on their contacts
    """
    trust_dict = {'R': attributes_rural_trust(), 'S': attributes_suburban_trust(), 'U': attributes_urban_trust(),
                  'BR': attributes_buch_rural_trust(), 'BS': attributes_buch_suburban_trust(),
                  'RR': attributes_ril_rural_trust(),
                  'RS': attributes_ril_suburban_trust(), 'RU': attributes_ril_urban_trust(),
                  'PS': attributes_pla_suburban_trust(),
                  'PU': attributes_pla_urban_trust()}
    return trust_dict


def source_interact_disaster():
    """
        :return: distribution of the contacts and frequency of interaction
    """
    source_interact_dict = {'R': attributes_rural_dis(), 'S': attributes_suburban_dis(), 'U': attributes_urban_dis(),
                            'BR': attributes_buch_rural_dis(), 'BS': attributes_buch_suburban_dis(),
                            'RR': attributes_ril_rural_dis(),
                            'RS': attributes_ril_suburban_dis(), 'RU': attributes_ril_urban_dis(),
                            'PS': attributes_pla_suburban_dis(),
                            'PU': attributes_pla_urban_dis()}
    return source_interact_dict


def trust_interact_disaster():
    """
        :return: trust that each survey respondant have on their contacts
    """
    trust_dict = {'R': attributes_rural_trust_dis(), 'S': attributes_suburban_trust_dis(),
                  'U': attributes_urban_trust_dis(),
                  'BR': attributes_buch_rural_trust_dis(), 'BS': attributes_buch_suburban_trust_dis(),
                  'RR': attributes_ril_rural_trust_dis(),
                  'RS': attributes_ril_suburban_trust_dis(), 'RU': attributes_ril_urban_trust_dis(),
                  'PS': attributes_pla_suburban_trust_dis(),
                  'PU': attributes_pla_urban_trust_dis()}
    return trust_dict


###New Disaster using Question 6

def disaster_data2():
    source_interact_dict = {'R': grouped2_rural['Q6_ND_people'], 'S': grouped2_suburban['Q6_ND_people'],
                            'U': grouped2_urban['Q6_ND_people'],
                            'BR': grouped1_buchanan_rural["Q6_ND_people"],
                            'BS': grouped1_buchanan_suburban['Q6_ND_people'],
                            'RR': grouped1_riley_rural['Q6_ND_people'],
                            'RS': grouped1_riley_suburban['Q6_ND_people'], 'RU': grouped1_riley_urban['Q6_ND_people'],
                            'PS': grouped1_platte_suburban['Q6_ND_people'],
                            'PU': grouped1_platte_urban['Q6_ND_people']}
    return source_interact_dict


# Updated code
feature_dict = {"People_Trust": [], "Frequency_interact_people": [], "Source_Trust": [],
                "Frequency_interact_source": []}


def assign_occur(col_data, nums):
    values = [1, 2, 3, 4, 5]
    range_vals = [(0, 0), (1, 2), (3, 4), (5, 7), (8, 12)]
    hrs = 24 * 7
    tv_data = col_data.to_list()
    tv_data = list(filter(lambda x: x >= 0, tv_data))
    # append_new_line("regression.txt", "Frequency of interaction with people")
    # append_new_line("regression.txt", str(tv_data))
    init_source_list = r_sample_distb(tv_data, nums)
    init_source_list = list(map(lambda x: 1 if i < 1 else x, init_source_list))
    init_source_list = list(map(lambda x: 5 if i > 5 else x, init_source_list))
    append_new_line("data/intermediate/regression.txt", "Frequency of interaction with people")
    append_new_line("data/intermediate/regression.txt", str(init_source_list))
    feature_dict["Frequency_interact_people"].append(init_source_list)
    # create dictionary of values and range
    source_dict = {}

    for i, j in zip(values, range_vals):
        source_dict[i] = j
    sampled = []
    for val in init_source_list:
        sources_val = source_dict[val]
        samp_val = random.randint(sources_val[0], sources_val[1])
        sampled.append(samp_val / hrs)
    return sampled


# Updated code
def sources_info(nums, col_data):
    values = [1, 2, 3, 4, 5]
    range_vals = [(0, 0), (1, 2), (3, 4), (5, 7), (8, 12)]
    hrs = 24 * 7
    tv_data = source_filter(col_data)
    tv_data = list(filter(lambda x: x >= 0, tv_data))

    init_source_list = r_sample_distb(tv_data, nums)
    init_source_list = list(map(lambda x: 1 if i < 1 else x, init_source_list))
    init_source_list = list(map(lambda x: 5 if i > 5 else x, init_source_list))
    append_new_line("data/intermediate/regression.txt", "Frequency of interaction with sources")
    append_new_line("data/intermediate/regression.txt", str(init_source_list))
    feature_dict["Frequency_interact_source"].append(init_source_list)
    # create dictionary of values and range

    source_dict = {}

    for i, j in zip(values, range_vals):
        source_dict[i] = j
    sampled = []
    for val in init_source_list:
        sources_val = source_dict[val]
        samp_val = random.randint(sources_val[0], sources_val[1])
        sampled.append(samp_val / hrs)

    return sampled


def get_neighbors(the_graph, initial_seed):
    """
    This is used to get the neighbors of a given node from the graph object.
    """
    target_neighbors = []
    for node in initial_seed:
        target_neighbors.append(list(the_graph.neighbors(node)))
    return target_neighbors


def create_graph(G):
    """
    Add state to the given graph
    #Removed for loop to get better performance
    """
    state = 0
    nx.set_node_attributes(G, state, "state")
    return G


def add_hubs(num=1):
    hubs = [random.randint(100, 160) for i in range(0, num)]
    return hubs


def add_seeds(G, maxi_node, init_len_nodes, seeds, col_val, news_trust=[]):
    seed_node = maxi_node + 1
    G.add_node(seed_node, state=1)

    # Add edge and edge weights between this TV node and all nodes in network
    seeds.append(seed_node)
    seed_weights = sources_info(init_len_nodes, col_val)

    if news_trust != []:
        trust_vals = r_sample_distb(news_trust, init_len_nodes)
        trust_vals = list(map(lambda x: 0 if x < 0 else x, trust_vals))
        trust_vals = list(map(lambda x: 10 if x > 10 else x, trust_vals))
        append_new_line("data/intermediate/regression.txt", "Trust of each source")
        append_new_line("data/intermediate/regression.txt", str(trust_vals))
        feature_dict["Source_Trust"].append(trust_vals)

    else:
        trust_vals = []

    if trust_vals == []:
        for i, k in zip(G.nodes(), seed_weights):
            if i in seeds:
                continue
            G.add_edge(seed_node, i, weight=k)
    else:
        for i, v, k in zip(G.nodes(), trust_vals, seed_weights):
            if i in seeds:
                continue
            G.add_edge(seed_node, i, trust=v, weight=k)
    return G, seed_node, seeds


def assign_trust_interactions(trust_list, nums):
    """
    :param trust_list: dataframe of trust values
    :param nums: number of nodes in the graph
    :return: graph G
    """
    init_pple_trust = r_sample_distb(trust_list, nums * 5)
    init_pple_trust = list(map(lambda i: 0 if i < 0 else i, init_pple_trust))
    init_pple_trust = list(map(lambda i: 10 if i > 10 else i, init_pple_trust))
    append_new_line("data/intermediate/regression.txt", "Trust of people")
    append_new_line("data/intermediate/regression.txt", str(init_pple_trust))
    feature_dict["People_Trust"].append(init_pple_trust)
    return init_pple_trust


def choose_random_edges(G):
    # convert to dicts of lists and select the neighbors
    arr = nx.to_dict_of_lists(G)
    graphlist = []
    for i in arr:
        try:
            sampled = random.sample(arr[i], 5)
            create_edges = [(i, j) for j in sampled]
            graphlist.extend(create_edges)
        except:
            sampled = arr[i]
            create_edges = [(i, j) for j in sampled]
            graphlist.extend(create_edges)
    return graphlist


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def simulation(G, sims, pop_percent, source, interact_val, discount, trust_int_list=[], news_trust=[]):
    """
    This is a function that carries out the information propagation
    G = networkx graph object
    sim = number of times this should be run
    """

    seeds = []
    init_len_nodes = G.number_of_nodes()
    maxi_node = list(G.nodes)[-1]
    nodes_in_graph = list(G.nodes)
    sampled_edges = choose_random_edges(G)

    # Add trust to interactions
    if trust_int_list != []:
        trust_int = assign_trust_interactions(trust_int_list, init_len_nodes)
        for x, y in zip(sampled_edges, trust_int):
            a, b = x
            G[a][b]["trust"] = y

        for u in G.edges(data=True):
            if not u[2]:
                G[u[0]][u[1]]["trust"] = 5

    # Add a TV node to the graph

    if isinstance(source, list):
        # print("news_trust",news_trust)
        for i, v in zip(source, news_trust):
            G, maxi_node, seeds = add_seeds(G, maxi_node, init_len_nodes, seeds, i, v)

    else:
        G, maxi_node, seeds = add_seeds(G, maxi_node, init_len_nodes, seeds, source, news_trust)

    edges_g = G.edges()

    rand_lambda = assign_occur(interact_val, len(sampled_edges))

    # Assign for occurrences
    # Correct this,apply to only 5 edges and use a constant for the rest
    # Select  random nodes for each node and assign this occurence
    for edge, v in zip(sampled_edges, rand_lambda):
        G.add_edge(edge[0], edge[1], weight=v)

    specific_edges = [i for i in edges_g(data=True) if 'weight' not in i[2]]  # list
    constant_val_occurrrence = [5 / (24 * 7)] * len(specific_edges)
    for edge, v in zip(specific_edges, constant_val_occurrrence):
        G.add_edge(edge[0], edge[1], weight=v)

    # probability of 0 occurrence,assign to all edges of the graph
    k = 0
    all_edge_data = [i[2]['weight'] for i in G.edges(data=True)]

    zero_occur_prob = [poisson_distribution(k, i) for i in all_edge_data]

    for u, z in zip(G.edges(), zero_occur_prob):
        (x, y) = u
        G[x][y]["zero_occur_prob"] = z

    # check for state that is 0, collect the neighbors of the state
    # propagate from not exposed to exposed

    # list with changing nodes at each time step
    total_time = []
    len_nodes = len(nodes_in_graph)
    pop_percent = int((pop_percent * len_nodes) / 100)

    if news_trust == []:
        trust = False
    else:
        trust = True
        # Add two attributes to graph. This stores changing attributes in graph during propagation
        nx.set_node_attributes(G, 'none', 'number_of_meets')
        nx.set_node_attributes(G, 'none', 'time_step')
        for i in G:
            G.nodes[i]["number_of_meets"] = {}
            G.nodes[i]["time_step"] = {}

    pool = mp.Pool(processes=1)
    func = functools.partial(sub_simulation, G, total_time, pop_percent, trust, discount)
    result = [pool.apply_async(func) for _ in range(sims)]
    res = [f.get() for f in result]
    pool.close()

    return res, feature_dict
