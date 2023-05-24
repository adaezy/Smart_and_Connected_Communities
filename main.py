import time
from scripts.assign_attributes import *
import click
from scripts.rkernel_estimate import r_sample_distb
import sys
from numpy import unique

def drop_missing_values(data):
    """
    Apply this to frequency of interactions
    :param data:
    :return:
    """
    new_data = np.delete(data,np.where(data==-99))
    # Delete these row indexes from dataFrame
    return new_data


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


@click.command()
@click.option('--disaster', help='Options are R,S,U,BR,BS,RR,RS,RU,PS,PU')

@click.option('--nodisaster', help="Options are 'R','S','U','BR','BS',RR,RS,RU,PS,PU")

@click.option('--disaster2')

@click.option('--trust')

@click.option('--social_net')  #For social network

@click.option('--government')

@click.option('--local_news')

@click.option('--print_news')

@click.option('--news')


###@click.argument('hub',default='')
###@click.argument('threshold',default = '')
def call_disaster_type(disaster, nodisaster, disaster2, trust,social_net,government,local_news,print_news,news):
    t1 = time.time()
    #my_pop = [1000, 5000, 10000, 15000, 20000, 25000, 50000]
    #my_pop = [50000, 25000, 20000, 15000, 10000, 5000, 1000]
    pop_size = 500 # 100, 200]#, 20000, 25000, 50000]
    source_of_info = []
    news_trust = []

    if disaster2:
        print("During disaster2 ...")
        s_int = source_interact_disaster()
        _, interactions = s_int[disaster2]
        distb = disaster_data2()[disaster2]
        if trust:
            print("This includes trust a user have for their 5 most trusted friends.")
            trust_values = trust_interact_disaster()
            pple_trust = trust_values[trust] #removed news_trust
        else:
            print("Trust is not enabled")
            pple_trust = []

        if social_net:
            print('Social network is enabled')
            social_nw_source = source_filter(disaster_social_nw()[social_net])
            source_of_info.append(social_nw_source)
            if trust:
                print("Social network trust is enabled")
                sn_trust = disaster_social_nw_trust()[trust]
                news_trust.append(sn_trust)
        if government:
            print('Government news is enabled')
            gov_nw_source = source_filter(disaster_gov_nw()[government])
            source_of_info.append(gov_nw_source)
            if trust:
                print("Government trust is enabled")
                gov_trust = disaster_gov_nw_trust()[trust]
                news_trust.append(gov_trust)
        if local_news:
            print('Local news is enabled')
            local_nw_source = source_filter(disaster_lcnews_nw()[local_news])
            source_of_info.append(local_nw_source)
            if trust:
                print("Local trust is enabled")
                local_trust = disaster_lcnews_nw_trust()[trust]
                news_trust.append(local_trust)
        if print_news:
            print('Print is enabled')
            print_news_nw_source = source_filter(disaster_print_nw()[print_news])
            source_of_info.append(print_news_nw_source)
            if trust:
                print("Print trust is enabled")
                print_trust = disaster_print_nw_trust()[trust]
                news_trust.append(print_trust)
        if news:
            print('News is enabled')
            news_nw_source = source_filter(disaster_cable_news_nw()[news])
            source_of_info.append(news_nw_source)
            if trust:
                print("News trust is enabled")
                trust_news = disaster_cable_news_nw_trust()[trust]
                news_trust.append(trust_news)

    if disaster:
        print("During disaster ...")
        s_int = source_interact_disaster()
        distb, interactions = s_int[disaster]
        #print(distb)
        #print(interactions)
        if trust:
            print("This includes trust  a user have for their trusted friends.")
            trust_values = trust_interact_disaster()
            pple_trust= trust_values[trust]
            #print(pple_trust)
        else:
            print("Trust is not enabled")
            pple_trust = []

        if social_net:
            print('Social network is enabled')
            social_nw_source = source_filter(disaster_social_nw()[social_net])
            #print(social_nw_source)
            source_of_info.append(social_nw_source)
            if trust:
                print("Social network trust is enabled")
                sn_trust = disaster_social_nw_trust()[trust]
                #print(sn_trust)
                news_trust.append(sn_trust)
        if government:
            print('Government news is enabled')
            gov_nw_source = source_filter(disaster_gov_nw()[government])
            #print(gov_nw_source)
            source_of_info.append(gov_nw_source)
            if trust:
                print("Government trust is enabled")
                gov_trust = disaster_gov_nw_trust()[trust]
                #print(gov_trust)
                news_trust.append(gov_trust)
        if local_news:
            print('Local news is enabled')
            local_nw_source = source_filter(disaster_lcnews_nw()[local_news])
            #print(local_nw_source)
            source_of_info.append(local_nw_source)
            if trust:
                print("Local trust is enabled")
                local_trust = disaster_lcnews_nw_trust()[trust]
                #print(local_trust)
                news_trust.append(local_trust)
        if print_news:
            print('Print is enabled')
            print_news_nw_source = source_filter(disaster_print_nw()[print_news])
            #print(print_news_nw_source)
            source_of_info.append(print_news_nw_source)
            if trust:
                print("Print trust is enabled")
                print_trust = disaster_print_nw_trust()[trust]
                #print(print_trust)
                news_trust.append(print_trust)
        if news:
            print('News is enabled')
            news_nw_source = source_filter(disaster_cable_news_nw()[news])
            #print(news_nw_source)
            source_of_info.append(news_nw_source)
            if trust:
                print("News trust is enabled")
                trust_news = disaster_cable_news_nw_trust()[trust]
                #print(trust_news)
                news_trust.append(trust_news)

    if nodisaster:
        print("During no disaster ...")
        s_int = source_interact_none()
        distb, interactions = s_int[nodisaster] #removed source
        #print(distb)
        #print(interactions)
        if trust:
            print("This includes trust  a user have for their trusted friends.")
            trust_values = trust_interact_none()
            pple_trust = trust_values[trust] #removed news_trust
            #print(pple_trust)
        else:
            print("Trust is not enabled")
            #news_trust = []
            pple_trust = []
        if social_net:
            print('Social network is enabled')
            social_nw_source = source_filter(nodisaster_social_nw()[social_net])
            #print(social_nw_source)
            source_of_info.append(social_nw_source)
            if trust:
                print("Social network trust is enabled")
                sn_trust = nodisaster_social_nw_trust()[trust]
                #print(sn_trust)
                news_trust.append(sn_trust)
        if government:
            print('Government news is enabled')
            gov_nw_source = source_filter(nodisaster_gov_nw()[government])
            #print(gov_nw_source)
            source_of_info.append(gov_nw_source)
            if trust:
                print("Government trust is enabled")
                gov_trust = nodisaster_gov_nw_trust()[trust]
                #print(gov_trust)
                news_trust.append(gov_trust)
        if local_news:
            print('Local news is enabled')
            local_nw_source = source_filter(nodisaster_lcnews_nw()[local_news])
            #print(local_nw_source)
            source_of_info.append(local_nw_source)
            if trust:
                print("Local trust is enabled")
                local_trust = nodisaster_lcnews_nw_trust()[trust]
                #print(local_trust)
                news_trust.append(local_trust)
        if print_news:
            print('Print is enabled')
            print_news_nw_source = source_filter(nodisaster_print_nw()[print_news])
            #print(print_news_nw_source)
            source_of_info.append(print_news_nw_source)
            if trust:
                print("Print trust is enabled")
                print_trust = nodisaster_print_nw_trust()[trust]
                #print(print_trust)
                news_trust.append(print_trust)
        if news:
            print('News is enabled')
            news_nw_source = source_filter(nodisaster_cable_news_nw()[news])
            #print(news_nw_source)
            source_of_info.append(news_nw_source)
            if trust:
                print("News trust is enabled")
                trust_news = nodisaster_cable_news_nw_trust()[trust]
                news_trust.append(trust_news)

    results=[]
    diffusion_dict = {}
    diffusion_dict["Graph_edges"]  = []
    diffusion_dict["Propagation Time"] = []
    diffusion_dict["Distribution"] = []

    #Save this values in a file
    print("distb", distb.to_dict())  # dict
    print("Frequency_source_of_info", source_of_info)  # list  of list
    print("Frequency of interactions", list(interactions.to_dict().values()))  # dict
    print("pple_trust", pple_trust)  # list
    print("news_trust", news_trust)  # list of list


    print("Results for",pop_size)
    print("\n")
    num = 0
    samples = r_sample_distb(distb, pop_size)
    samples = list(map(lambda k: 0 if k < 0 else k, samples))

    dict_of_times = {}
    threshold = [30] #[10, 20, 30, 40, 50]

    for x in range(0,100):
        print("Graph",x+1)
        num += 1

        g = create_graph(configuration_model(samples))

        append_new_line("data/intermediate/distribution_results.txt", "\n" + str(num) + ".")
        append_new_line("data/intermediate/distribution_results.txt", "Graph Edges\n:" + str(g.degree()))
        append_new_line("data/intermediate/distribution_results.txt", "Graph_diff\n:" + str(round(nx.degree_pearson_correlation_coefficient(g), 4)))


        for dis in threshold:
            h = g.copy()

            #Number should be 1000  simulations
            total_time,feature_dicts = simulation(h, 1000, 90, source_of_info, interactions,dis,pple_trust,news_trust)

            total_time = flatten_lofl(total_time)
            print("total_time",total_time)
            values, counts = unique(total_time, return_counts=True)
            results.append(values.tolist())
            results.append(counts.tolist())

            append_new_line("data/intermediate/distribution_results.txt", "Propagation Time:" + str(values.tolist()))
            append_new_line("data/intermediate/distribution_results.txt", "Distribution:" + str(counts.tolist()))
            t2 = time.time()
            append_new_line("data/intermediate/distribution_results.txt", "Time:" + str(t2 - t1))
            diffusion_dict["Graph_edges"].append(g.edges)
            diffusion_dict["Propagation Time"].append(values.tolist())
            diffusion_dict["Distribution"].append(counts.tolist())
            avg_time_diffusion = sum(total_time)/len(total_time)
            #print("Average Time of Diffusion for 1000 iterations",avg_time_diffusion )
            if dis not in dict_of_times:
                dict_of_times[dis] = [avg_time_diffusion]
            else:
                dict_of_times[dis].append(avg_time_diffusion)
            print(dict_of_times)
    #Calculate average time of diffusion  and std deviation for 100 graphs
    final_avg ={}
    final_std ={}
    for tm in dict_of_times:
        final_avg[tm] = sum(dict_of_times[tm])/len(dict_of_times[tm])
        final_std[tm] = round(np.std(dict_of_times[tm], ddof=1),3)
    print("Average Time of Diffusion for 100 graphs",final_avg)
    print("Standard Deviation",final_std)

def main():
    if sys.argv[1] not in ['--disaster', '--nodisaster', '--disaster2']:
        print("Expected arguments --disaster or --nodisaster or --disaster2 ")
    call_disaster_type()

if __name__ == '__main__':
    main()

