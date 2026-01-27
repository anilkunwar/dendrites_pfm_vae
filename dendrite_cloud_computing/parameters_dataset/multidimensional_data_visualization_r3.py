import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Critical for Streamlit compatibility
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle, Wedge
from matplotlib.collections import PatchCollection, LineCollection
from matplotlib.patches import FancyBboxPatch
import matplotlib.colors as mcolors
from matplotlib.colorbar import Colorbar
import io
import warnings
from scipy import stats
import traceback
warnings.filterwarnings('ignore')

# ==================== HARD-CODED EXAMPLE DATA ====================
EXAMPLE_CSV = """step,score,coverage,hopping_strength,t,POT_LEFT,fo,Al,Bl,Cl,As,Bs,Cs,cleq,cseq,L1o,L2o,ko,Noise
0,5.169830808799158,0.10416666666666667,0.1,0.028256090357899666,0.4339944124221802,0.29827773571014404,0.5466359853744507,0.49647387862205505,0.4251793622970581,0.5390684008598328,0.33983609080314636,0.5945582389831543,0.46486878395080566,0.4681702256202698,0.359584778547287,0.6078763008117676,0.4093511402606964,0.31751325726509094
1,5.179830808799158,0.10460069444444445,0.11139433523068368,0.0594908632338047,0.4270627200603485,0.27864405512809753,0.5769832134246826,0.5266308188438416,0.36398613452911377,0.6301330327987671,0.35744181275367737,0.5121908187866211,0.3641984462738037,0.4810342490673065,0.37694957852363586,0.612697958946228,0.4058190882205963,0.34840521216392517
2,5.304021017662722,0.10460069444444445,0.12041199826559248,0.08961087465286255,0.4117993414402008,0.3086051940917969,0.5759025812149048,0.5457133650779724,0.37897956371307373,0.6828550696372986,0.37128087878227234,0.5426892042160034,0.31361812353134155,0.5300193428993225,0.3330453932285309,0.6396515965461731,0.4162127375602722,0.386873722076416
3,5.169830808799158,0.11197916666666667,0.1278753600952829,0.11320602893829346,0.5374900102615356,0.3893888592720032,0.6208136677742004,0.5366111993789673,0.4114663004875183,0.7501887679100037,0.4020015299320221,0.45790043473243713,0.343185693025589,0.7146779298782349,0.3623429834842682,0.6474707126617432,0.4218909740447998,0.414340615272522
4,5.169830808799158,0.11371527777777778,0.13424226808222062,0.1594301164150238,0.5966717004776001,0.4405992031097412,0.593701183795929,0.6432862877845764,0.48903852701187134,0.8087910413742065,0.4361550211906433,0.5463976263999939,0.351766973733902,0.7200356721878052,0.39372894167900085,0.699080228805542,0.46395444869995117,0.33751198649406433
5,5.1838690114837185,0.12890625,0.13979400086720378,0.27053752541542053,0.5414740443229675,0.3837757110595703,0.5459027290344238,0.7114285826683044,0.5601555109024048,0.8636823296546936,0.4749370515346527,0.5336172580718994,0.49408119916915894,0.6982162594795227,0.4363292157649994,0.6871276497840881,0.5171192288398743,0.34152668714523315
6,5.100333034308876,0.1440972222222222,0.14471580313422192,0.3258400559425354,0.6228137016296387,0.49862906336784363,0.6784278154373169,0.8143172264099121,0.5625902414321899,0.896495521068573,0.6146658062934875,0.5892767310142517,0.4974607229232788,0.8461188077926636,0.5145464539527893,0.8220916986465454,0.6068583726882935,0.40622031688690186
7,7.32416026628129,0.1840277777777778,0.14913616938342728,0.3501424491405487,0.3400475084781647,0.5205002427101135,0.697181224822998,0.9804031848907471,0.6573023200035095,0.8734657168388367,0.8090043663978577,0.7597450017929077,0.57285475730896,0.7757821083068848,0.645453929901123,0.8650208711624146,0.5720763802528381,0.4552328288555145
8,10.386888918380697,0.22178819444444445,0.15314789170422552,0.377905011177063,0.3762165307998657,0.5274438858032227,0.6071285009384155,0.9385737180709839,0.6077378988265991,0.8730340600013733,0.9395252466201782,0.7045806646347046,0.5667017102241516,0.7497444152832031,0.8033575415611267,0.7816545963287354,0.5264078378677368,0.5300353765487671
9,8.254108578194927,0.3077256944444444,0.1568201724066995,0.3872142434120178,0.47770050168037415,0.4989684224128723,0.7149577140808105,1.0060926675796509,0.6737095713615417,0.7603132724761963,0.6836521625518799,0.6091089844703674,0.4301818609237671,0.8038907051086426,0.7902536988258362,0.9215232133865356,0.5898656249046326,0.6257796287536621
10,5.622075167395693,0.3493923611111111,0.16020599913279623,0.3937322199344635,0.33547544479370117,0.44358500838279724,0.7463698983192444,0.8410062193870544,0.6607444286346436,0.6670836806297302,0.6114344596862793,0.6689836382865906,0.4694424569606781,0.7656615376472473,0.7240639328956604,0.8651359677314758,0.7883168458938599,0.6361547112464905
11,7.321904299308076,0.3650173611111111,0.16334684555795864,0.45904433727264404,0.27944284677505493,0.38179677724838257,0.8159457445144653,1.0150943994522095,0.7721518874168396,0.7039114236831665,0.7788766026496887,0.7146820425987244,0.49762389063835144,0.7908686399459839,0.8121711611747742,0.8742984533309937,0.7975523471832275,0.6416141390800476
12,8.543683990691765,0.3723958333333333,0.1662757831681574,0.5105248689651489,0.34250542521476746,0.43922367691993713,0.7341154217720032,1.1102296113967896,0.7483778595924377,0.7128080725669861,0.7358441352844238,0.7393836975097656,0.46837639808654785,0.8202276825904846,0.877817690372467,0.9840682744979858,0.7861328125,0.6163419485092163
13,7.620555709528761,0.3732638888888889,0.16901960800285137,0.5417518019676208,0.43418756127357483,0.40142834186553955,0.7275840044021606,1.0893441438674927,0.7854703664779663,0.7538084983825684,0.6650506258010864,0.796431303024292,0.4985898435115814,0.8372722864151001,0.744591236114502,1.123425841331482,0.785809338092804,0.680273711681366
14,6.3356596618964565,0.3745659722222222,0.17160033436347993,0.5423417687416077,0.26539722084999084,0.5920475721359253,0.6683412790298462,1.082399845123291,0.6814282536506653,0.8131998181343079,0.6773099303245544,0.7042008638381958,0.5497509837150574,0.6867591738700867,0.8954933881759644,0.8905871510505676,0.7955514192581177,0.6689854264259338
15,7.436352609042917,0.40625,0.17403626894942437,0.568615198135376,0.24044141173362732,0.6975681781768799,0.6411821246147156,1.1813530921936035,0.7024039626121521,0.8560113310813904,0.709820568561554,0.76691734790802,0.6928378343582153,0.4951046407222748,0.9582697153091431,0.9249935150146484,0.7339156866073608,0.7971505522727966
16,10.277262759911848,0.4618055555555556,0.17634279935629374,0.5763835906982422,0.46326711773872375,0.8927175998687744,0.6113301515579224,1.1488991975784302,0.8268088698387146,0.8452023863792419,0.6038599610328674,0.7312695980072021,0.7745003700256348,0.523252010345459,0.9515889286994934,0.7617896795272827,0.7811905741691589,0.7240396738052368
17,10.171294809365978,0.46484375,0.17853298350107671,0.6809343695640564,0.5394753217697144,0.7823241353034973,0.5580039620399475,1.2827746868133545,0.8729355335235596,0.9322726130485535,0.4856688380241394,0.942378580570221,0.7398176789283752,0.6016049981117249,0.8035934567451477,0.7253121137619019,0.6769372820854187,0.6313362121582031
18,10.629263728382883,0.48046875,0.1806179973983887,0.6865774393081665,0.5340346097946167,0.8711283802986145,0.6346777677536011,1.232739806175232,0.9320783019065857,0.9418307542800903,0.49510204792022705,1.0256969928741455,0.7569521069526672,0.6322088837623596,0.8393365144729614,0.7755464911460876,0.706352949142456,0.6519243717193604
19,10.49475562025519,0.4822048611111111,0.18260748027008264,0.7152098417282104,0.4992449879646301,0.9342008233070374,0.8139623403549194,1.3508509397506714,1.0962871313095093,0.9878286719322205,0.6010639071464539,0.9243364930152893,0.9159383177757263,0.6504509449005127,0.913641095161438,0.8966460227966309,0.9222412705421448,0.878772497177124
20,10.22933457672259,0.4887152777777778,0.1845098040014257,0.7465022206306458,0.5899507403373718,0.9446436762809753,0.8453498482704163,1.564751148223877,1.0583618879318237,1.1709020137786865,0.7551482319831848,0.9316735863685608,1.0114786624908447,0.7386069893836975,0.8721351027488708,0.712131679058075,0.9934794902801514,0.8559936881065369
21,10.40191081814774,0.5212673611111112,0.1863322860120456,0.778833270072937,0.46530479192733765,0.9134830236434937,0.8916224837303162,1.5924957990646362,1.0132859945297241,1.3060201406478882,0.7646929621696472,1.0435600280761719,0.8411058783531189,0.7712708711624146,0.9179795384407043,0.8546093702316284,0.9429539442062378,0.8560317158699036
22,11.174144433501233,0.5386284722222222,0.18808135922807911,0.8746334314346313,0.48324069380760193,0.9938526749610901,0.769467830657959,1.6170459985733032,1.01220703125,1.2132935523986816,0.752862811088562,1.16727614402771,0.9052226543426514,0.6134433150291443,0.9348587393760681,0.8925090432167053,0.8277872204780579,0.9739061594009399
23,10.844935844038055,0.5503472222222222,0.18976270912904414,0.9238934516906738,0.5506340861320496,0.9444273114204407,0.8249781131744385,1.7516707181930542,0.8675052523612976,1.4430488348007202,0.8421710729598999,1.072837471961975,1.066247820854187,0.9110193252563477,0.8453015089035034,0.6934573650360107,0.8824711441993713,0.9735444188117981
24,9.859185254537385,0.5525173611111112,0.19138138523837167,0.9569121599197388,0.6011497378349304,1.018949031829834,0.8612451553344727,1.6675461530685425,0.9851047396659851,1.3370230197906494,0.9144415855407715,1.1276476383209229,1.0655651092529297,0.8532272577285767,0.8896656036376953,0.7149348855018616,0.8930221796035767,1.089601993560791
25,9.733399429950971,0.5572916666666666,0.19294189257142927,0.9699379205703735,0.5459413528442383,0.9786086082458496,0.9722445607185364,1.618293046951294,1.0138611793518066,1.2777221202850342,0.769503653049469,1.2075700759887695,0.9359092116355896,0.8779188990592957,0.8745647072792053,0.5999792814254761,0.8872678279876709,1.0222569704055786
26,9.9260127892421,0.5690104166666666,0.19444826721501687,1.0167112350463867,0.7404793500900269,1.2331862449645996,0.9591841697692871,1.8226509094238281,1.1032135486602783,1.5929903984069824,1.0649257898330688,1.510267734527588,0.9941714406013489,1.090094804763794,0.8625853061676025,0.5799097418785095,0.9083585143089294,0.8630198836326599
27,10.678004251308256,0.5798611111111112,0.19590413923210936,1.1897140741348267,0.8140103220939636,1.1114305257797241,0.8603397011756897,1.7859822511672974,1.1018062829971313,1.5066109895706177,0.9222522377967834,1.4768295288085938,0.9960513710975647,0.8831607103347778,0.8123284578323364,0.8251258730888367,0.7022106647491455,0.8924230933189392
28,11.020299291797908,0.5798611111111112,0.19731278535996988,1.1938612461090088,0.8728200793266296,1.2563399076461792,0.9425970911979675,1.665657877922058,1.1347849369049072,1.6763732433319092,0.9476985335350037,1.450101613998413,0.9543245434761047,0.7781178951263428,0.7439810633659363,0.7765737175941467,0.9004281163215637,0.8755770325660706
29,11.988932619625839,0.5911458333333334,0.19867717342662447,1.2421698570251465,0.9625719785690308,1.3612264394760132,0.8735606670379639,1.6014152765274048,1.1779128313064575,1.7611147165298462,0.9013581275939941,1.336272954940796,0.8664807081222534,0.7415759563446045,0.8296802639961243,0.8429422974586487,1.0249007940292358,0.7863231301307678
30,11.784495105579659,0.5933159722222222,0.2,1.3608571290969849,0.9625561833381653,1.2631572484970093,0.9952393770217896,1.82819402217865,1.1884130239486694,1.8113062381744385,0.981305718421936,1.5592151880264282,0.9318991899490356,0.6798413395881653,0.864549994468689,0.8141895532608032,0.9140636324882507,0.9269731044769287"""

# ==================== EXTENSIVE COLORMAP LISTS ====================
ALL_COLORMAPS = sorted(set([
    'jet', 'rainbow', 'turbo', 'inferno', 'plasma', 'viridis', 'magma', 'cividis',
    'hot', 'cool', 'hot_r', 'cool_r', 'spring', 'summer', 'autumn', 'winter',
    'bone', 'copper', 'pink', 'gray', 'spectral', 'gist_rainbow', 'rainbow_r',
    'nipy_spectral', 'gist_ncar', 'gist_stern', 'flag', 'prism', 'ocean', 'gist_earth',
    'terrain', 'gist_stern_r', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix',
    'brg', 'hsv', 'gist_rainbow_r', 'seismic', 'coolwarm', 'bwr', 'RdBu', 'RdGy',
    'PiYG', 'PRGn', 'RdYlBu', 'RdYlGn', 'Spectral', 'twilight', 'twilight_shifted',
    'hsv_r', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3',
    'tab10', 'tab20', 'tab20b', 'tab20c', 'flag_r', 'prism_r', 'ocean_r', 'gist_earth_r',
    'terrain_r', 'gist_stern_r', 'gnuplot_r', 'gnuplot2_r', 'CMRmap_r', 'cubehelix_r',
    'brg_r', 'pink_r', 'binary', 'binary_r', 'gist_yarg', 'gist_yarg_r', 'afmhot',
    'afmhot_r', 'bone_r', 'copper_r', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges',
    'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu',
    'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'Greys_r', 'Purples_r', 'Blues_r', 'Greens_r',
    'Oranges_r', 'Reds_r', 'YlOrBr_r', 'YlOrRd_r', 'OrRd_r', 'PuRd_r', 'RdPu_r',
    'BuPu_r', 'GnBu_r', 'PuBu_r', 'YlGnBu_r', 'PuBuGn_r', 'BuGn_r', 'YlGn_r',
    'PiYG_r', 'PRGn_r', 'RdYlBu_r', 'RdYlGn_r', 'Spectral_r', 'coolwarm_r', 'bwr_r',
    'seismic_r', 'twilight_r', 'twilight_shifted_r', 'Set1_r', 'Set2_r', 'Set3_r',
    'Dark2_r', 'Accent_r', 'Paired_r', 'Pastel1_r', 'Pastel2_r', 'tab10_r', 'tab20_r',
    'tab20b_r', 'tab20c_r', 'Vega10', 'Vega20', 'Vega10_r', 'Vega20_r', 'Vega20b',
    'Vega20c', 'Vega20b_r', 'Vega20c_r', 'magma_r', 'inferno_r', 'plasma_r', 'viridis_r',
    'cividis_r', 'turbo_r', 'nipy_spectral_r', 'gist_ncar_r'
]))

# Statistical colormaps for correlations/covariances
STAT_COLORMAPS = ['RdBu', 'RdGy', 'coolwarm', 'bwr', 'seismic', 'PiYG', 'PRGn',
                  'RdYlBu', 'RdYlGn', 'Spectral', 'BrBG', 'PuOr']

# ==================== FEATURE CONFIGURATION ====================
FEATURE_COLS = ['t','POT_LEFT','fo','Al','Bl','Cl','As','Bs','Cs','cleq','cseq','L1o','L2o','ko','Noise']
NODE_NAMES = FEATURE_COLS

# ==================== FIXED CHORD DIAGRAM FUNCTION ====================
def create_correlation_chord_diagram(df_selected, colormap_name='RdBu',
                                     metric='correlation', label_size=12,
                                     node_size=100, edge_width_scale=3,
                                     edge_threshold=0.3, show_node_values=True,
                                     show_edge_values=False, normalize_nodes=True,
                                     figsize=(14, 14), title_prefix=""):
    """
    Create a chord diagram with features on circumference and correlations/covariances as connections
    
    Parameters:
    -----------
    df_selected : DataFrame
        DataFrame containing selected rows and feature columns
    colormap_name : str
        Name of matplotlib colormap for edges
    metric : str
        'correlation' or 'covariance'
    label_size : int
        Font size for labels
    node_size : float
        Base size for nodes
    edge_width_scale : float
        Scaling factor for edge widths
    edge_threshold : float
        Minimum absolute value to show edge
    show_node_values : bool
        Whether to show node values (means)
    show_edge_values : bool
        Whether to show edge values on chords
    normalize_nodes : bool
        Whether to normalize node sizes
    figsize : tuple
        Figure size (width, height)
    title_prefix : str
        Prefix for the title
    
    Returns:
    --------
    matplotlib.figure.Figure
        The chord diagram figure
    """
    # Calculate statistics
    feature_means = df_selected[FEATURE_COLS].mean().values
    feature_stds = df_selected[FEATURE_COLS].std().values
    
    # Calculate correlation or covariance matrix
    if metric == 'correlation':
        connection_matrix = df_selected[FEATURE_COLS].corr().fillna(0).values
        metric_label = "Correlation"
        vmin, vmax = -1, 1
        center = 0
    else:  # covariance
        connection_matrix = df_selected[FEATURE_COLS].cov().fillna(0).values
        metric_label = "Covariance"
        # Get symmetric bounds for colormap
        max_abs = np.max(np.abs(connection_matrix))
        vmin, vmax = -max_abs, max_abs
        center = 0
    
    n_features = len(FEATURE_COLS)
    
    # Calculate angles for node positions
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
    angles = np.roll(angles, -1)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=figsize)
    
    # Main chord diagram (polar plot)
    ax_main = plt.subplot(111, projection='polar')
    ax_main.set_theta_zero_location("N")
    ax_main.set_theta_direction(-1)
    ax_main.axis('off')
    
    # Get colormap for edges
    edge_cmap = cm.get_cmap(colormap_name)
    
    # Calculate node sizes based on feature means
    if normalize_nodes:
        if feature_means.max() > 0:
            node_sizes = (feature_means / feature_means.max()) * node_size
        else:
            node_sizes = np.ones(n_features) * node_size / 2
    else:
        node_sizes = np.ones(n_features) * node_size
    
    # Draw nodes (features on circumference)
    width = 2 * np.pi / n_features * 0.8
    bottom = 1.5
    
    # Get a categorical colormap for nodes
    node_cmap = cm.get_cmap('tab20', n_features)
    
    # Create node patches with gradient colors based on value
    node_patches = []
    for idx, (angle, size) in enumerate(zip(angles, node_sizes)):
        # Create gradient color from light to dark based on normalized value
        norm_val = feature_means[idx] / feature_means.max() if feature_means.max() > 0 else 0.5
        base_color = node_cmap(idx)
        
        # Adjust brightness based on value
        if norm_val > 0.5:
            node_color = tuple(min(1, c * (0.7 + 0.3*norm_val)) for c in base_color[:3]) + (0.9,)
        else:
            node_color = tuple(min(1, c * (0.4 + 0.6*norm_val)) for c in base_color[:3]) + (0.9,)
        
        # Create wedge for node
        wedge = Wedge((0, 0), bottom + size/2,
                      np.degrees(angle - width/2),
                      np.degrees(angle + width/2),
                      width=size/2,
                      facecolor=node_color,
                      edgecolor='white',
                      linewidth=2,
                      alpha=0.9)
        ax_main.add_patch(wedge)
        node_patches.append(wedge)
        
        # Add feature name label
        rotation = np.degrees(angle)
        if 90 <= rotation <= 270:
            rotation += 180
            align = "right"
        else:
            align = "left"
        
        label_y_offset = bottom + size + 1.5
        ax_main.text(
            angle,
            label_y_offset,
            NODE_NAMES[idx],
            ha=align,
            va='center',
            fontsize=label_size,
            fontweight='bold',
            rotation=rotation,
            rotation_mode='anchor',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8, edgecolor='gray')
        )
        
        # Add mean value label if requested
        if show_node_values:
            val_text = f"{feature_means[idx]:.3f}"
            ax_main.text(
                angle,
                bottom + size/2,
                val_text,
                ha='center',
                va='center',
                fontsize=max(8, label_size - 4),
                fontweight='normal',
                color='black',
                rotation=rotation,
                rotation_mode='anchor'
            )
    
    # Draw edges (connections) based on correlation/covariance
    # First, create a list of all connections
    connections = []
    edge_colors = []
    edge_widths = []
    edge_values = []
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            value = connection_matrix[i, j]
            abs_value = abs(value)
            
            # Skip weak connections below threshold
            if abs_value < edge_threshold:
                continue
            
            # Calculate edge properties
            edge_width = abs_value * edge_width_scale
            edge_color = edge_cmap((value - vmin) / (vmax - vmin))
            
            connections.append((i, j))
            edge_colors.append(edge_color)
            edge_widths.append(edge_width)
            edge_values.append(value)
    
    # Sort connections by absolute value for proper z-ordering
    sorted_indices = np.argsort([abs(v) for v in edge_values])
    connections = [connections[i] for i in sorted_indices]
    edge_colors = [edge_colors[i] for i in sorted_indices]
    edge_widths = [edge_widths[i] for i in sorted_indices]
    edge_values = [edge_values[i] for i in sorted_indices]
    
    # Draw connections with CORRECTED Path construction
    for (i, j), color, width, value in zip(connections, edge_colors, edge_widths, edge_values):
        # Get angles for the two nodes
        theta1, theta2 = angles[i], angles[j]
        
        # Calculate radii for chord path
        radius1 = bottom + node_sizes[i] / 2
        radius2 = bottom + node_sizes[j] / 2
        avg_radius = (radius1 + radius2) / 2
        
        # Create control points for Bezier curve
        control_radius = avg_radius * 0.7
        control_theta = (theta1 + theta2) / 2
        
        # FIXED: Proper quadratic Bezier path construction
        # Create two separate paths for the ribbon edges
        offset = width / 15
        
        # Top edge of ribbon
        verts_top = [
            (theta1, radius1),
            (control_theta, control_radius),
            (theta2, radius2)
        ]
        codes_top = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        
        # Bottom edge of ribbon (offset)
        verts_bottom = [
            (theta1, radius1 + offset),
            (control_theta, control_radius + offset),
            (theta2, radius2 + offset)
        ]
        codes_bottom = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        
        # Combine to create closed ribbon path
        verts = verts_top + verts_bottom[::-1] + [verts_top[0]]
        codes = codes_top + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
        
        path = Path(verts, codes)
        patch = PathPatch(path,
                         facecolor=color,
                         edgecolor='none',
                         alpha=0.7,
                         zorder=1)  # Lower zorder so nodes appear on top
        ax_main.add_patch(patch)
        
        # Add edge value label if requested
        if show_edge_values and abs(value) > edge_threshold * 1.5:
            label_angle = (theta1 + theta2) / 2
            label_radius = control_radius * 0.9
            ax_main.text(
                label_angle,
                label_radius,
                f"{value:.2f}",
                ha='center',
                va='center',
                fontsize=max(7, label_size - 6),
                fontweight='bold',
                color='white' if abs(value) > 0.5 else 'black',
                bbox=dict(boxstyle="round,pad=0.1",
                         facecolor=color,
                         alpha=0.8,
                         edgecolor='none')
            )
    
    # Add center circle with metric info
    center_circle = Circle((0, 0), bottom - 1,
                          facecolor='white',
                          edgecolor='gray',
                          linewidth=1,
                          alpha=0.9)
    ax_main.add_patch(center_circle)
    
    # Add metric info in center
    ax_main.text(0, 0, f"{metric_label}\nMatrix",
                ha='center', va='center',
                fontsize=label_size + 2,
                fontweight='bold')
    ax_main.text(0, -0.3, f"{len(df_selected)} rows",
                ha='center', va='center',
                fontsize=label_size - 2)
    
    # Set title
    plt.suptitle(f"{title_prefix}{metric_label} Chord Diagram",
                fontsize=label_size + 8,
                fontweight='bold',
                y=0.95)
    
    # Create a separate axes for colorbars and statistics
    fig.subplots_adjust(right=0.85, left=0.05)
    
    # Add edge colorbar
    cax_edges = fig.add_axes([0.88, 0.55, 0.03, 0.35])
    norm_edges = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm_edges = cm.ScalarMappable(norm=norm_edges, cmap=edge_cmap)
    sm_edges.set_array([])
    cbar_edges = fig.colorbar(sm_edges, cax=cax_edges, orientation='vertical')
    cbar_edges.set_label(f'{metric_label} Value', fontsize=label_size)
    cbar_edges.ax.tick_params(labelsize=label_size - 2)
    
    # Add node colorbar (showing feature indices)
    cax_nodes = fig.add_axes([0.88, 0.15, 0.03, 0.25])
    norm_nodes = mcolors.Normalize(vmin=0, vmax=n_features-1)
    sm_nodes = cm.ScalarMappable(norm=norm_nodes, cmap=node_cmap)
    sm_nodes.set_array([])
    cbar_nodes = fig.colorbar(sm_nodes, cax=cax_nodes, orientation='vertical')
    cbar_nodes.set_label('Feature Index', fontsize=label_size)
    cbar_nodes.set_ticks(np.arange(n_features))
    cbar_nodes.set_ticklabels(range(n_features))
    cbar_nodes.ax.tick_params(labelsize=label_size - 2)
    
    # Add statistics text box
    ax_stats = fig.add_axes([0.02, 0.02, 0.4, 0.15])
    ax_stats.axis('off')
    
    # Calculate statistics
    total_connections = len(connections)
    if total_connections > 0:
        avg_corr = np.mean([abs(v) for v in edge_values])
        max_corr = np.max([abs(v) for v in edge_values])
        min_corr = np.min([abs(v) for v in edge_values])
        positive_connections = sum(1 for v in edge_values if v > 0)
        negative_connections = sum(1 for v in edge_values if v < 0)
    else:
        avg_corr = max_corr = min_corr = 0
        positive_connections = negative_connections = 0
    
    stats_text = [
        f"Statistical Summary:",
        f"Features: {n_features}",
        f"Rows Analyzed: {len(df_selected)}",
        f"Connections Shown: {total_connections}",
        f"  • Positive: {positive_connections}",
        f"  • Negative: {negative_connections}",
        f"Strength Range: [{min_corr:.3f}, {max_corr:.3f}]",
        f"Average |{metric}|: {avg_corr:.3f}"
    ]
    
    stats_y = 0.95
    for line in stats_text:
        ax_stats.text(0.02, stats_y, line,
                     fontsize=label_size - 2,
                     verticalalignment='top',
                     transform=ax_stats.transAxes)
        stats_y -= 0.12
    
    plt.tight_layout()
    return fig

# ==================== MULTI-LAYER CHORD DIAGRAM FUNCTION ====================
def create_multi_layer_chord_diagram(df_selected, colormap_corr='RdBu',
                                     colormap_cov='coolwarm', label_size=12,
                                     node_size=100, figsize=(16, 12),
                                     show_both_matrices=True):
    """
    Create a multi-layer chord diagram showing both correlation and covariance
    
    Parameters:
    -----------
    df_selected : DataFrame
        DataFrame containing selected rows
    colormap_corr : str
        Colormap for correlation matrix
    colormap_cov : str
        Colormap for covariance matrix
    label_size : int
        Font size for labels
    node_size : float
        Base size for nodes
    figsize : tuple
        Figure size (width, height)
    show_both_matrices : bool
        Whether to show both matrices or just correlation
    
    Returns:
    --------
    matplotlib.figure.Figure
        The multi-layer chord diagram figure
    """
    # Calculate matrices
    correlation_matrix = df_selected[FEATURE_COLS].corr().fillna(0).values
    covariance_matrix = df_selected[FEATURE_COLS].cov().fillna(0).values
    feature_means = df_selected[FEATURE_COLS].mean().values
    
    n_features = len(FEATURE_COLS)
    
    # Create figure with multiple subplots
    if show_both_matrices:
        fig, axes = plt.subplots(1, 3, figsize=figsize,
                                subplot_kw=dict(projection='polar'),
                                gridspec_kw={'width_ratios': [2, 2, 1]})
        ax_corr, ax_cov, ax_stats = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=(figsize[0]*0.75, figsize[1]),
                                subplot_kw=dict(projection='polar'),
                                gridspec_kw={'width_ratios': [3, 1]})
        ax_corr, ax_stats = axes
        ax_cov = None
    
    # Setup common parameters
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
    angles = np.roll(angles, -1)
    bottom = 1.5
    
    # Normalize node sizes
    if feature_means.max() > 0:
        node_sizes = (feature_means / feature_means.max()) * node_size
    else:
        node_sizes = np.ones(n_features) * node_size / 2
    
    width = 2 * np.pi / n_features * 0.7
    
    # Get colormaps
    cmap_corr = cm.get_cmap(colormap_corr)
    cmap_cov = cm.get_cmap(colormap_cov)
    node_cmap = cm.get_cmap('tab20', n_features)
    
    # Function to create a single chord diagram
    def create_single_diagram(ax, matrix, cmap, title, vmin=None, vmax=None):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.axis('off')
        
        # Set vmin/vmax for colormap
        if vmin is None:
            vmin = -1 if title == "Correlation" else -np.max(np.abs(matrix))
        if vmax is None:
            vmax = 1 if title == "Correlation" else np.max(np.abs(matrix))
        
        # Draw nodes
        for idx, (angle, size) in enumerate(zip(angles, node_sizes)):
            node_color = node_cmap(idx)
            wedge = Wedge((0, 0), bottom + size/2,
                         np.degrees(angle - width/2),
                         np.degrees(angle + width/2),
                         width=size/2,
                         facecolor=node_color,
                         edgecolor='white',
                         linewidth=1,
                         alpha=0.8)
            ax.add_patch(wedge)
            
            # Add labels only on first diagram
            if ax == ax_corr:
                rotation = np.degrees(angle)
                if 90 <= rotation <= 270:
                    rotation += 180
                    align = "right"
                else:
                    align = "left"
                ax.text(angle, bottom + size + 2, NODE_NAMES[idx],
                       ha=align, va='center',
                       fontsize=label_size, fontweight='bold',
                       rotation=rotation, rotation_mode='anchor')
        
        # Draw connections
        edge_threshold = 0.3 if title == "Correlation" else np.percentile(np.abs(matrix), 70)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                value = matrix[i, j]
                abs_value = abs(value)
                
                if abs_value < edge_threshold:
                    continue
                
                # Calculate edge properties
                edge_width = abs_value * 3
                edge_color = cmap((value - vmin) / (vmax - vmin))
                
                # Draw chord with CORRECTED path
                theta1, theta2 = angles[i], angles[j]
                radius1 = bottom + node_sizes[i] / 2
                radius2 = bottom + node_sizes[j] / 2
                control_radius = (radius1 + radius2) / 2 * 0.7
                control_theta = (theta1 + theta2) / 2
                
                offset = edge_width / 15
                
                # Top edge
                verts_top = [
                    (theta1, radius1),
                    (control_theta, control_radius),
                    (theta2, radius2)
                ]
                codes_top = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                
                # Bottom edge
                verts_bottom = [
                    (theta1, radius1 + offset),
                    (control_theta, control_radius + offset),
                    (theta2, radius2 + offset)
                ]
                codes_bottom = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
                
                # Combine
                verts = verts_top + verts_bottom[::-1] + [verts_top[0]]
                codes = codes_top + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
                
                path = Path(verts, codes)
                patch = PathPatch(path, facecolor=edge_color, edgecolor='none', alpha=0.6)
                ax.add_patch(patch)
        
        # Add center info
        center_circle = Circle((0, 0), bottom - 0.8,
                              facecolor='white', edgecolor='gray',
                              linewidth=1, alpha=0.9)
        ax.add_patch(center_circle)
        ax.text(0, 0, title[:4], ha='center', va='center',
               fontsize=label_size + 2, fontweight='bold')
        ax.set_title(title, fontsize=label_size + 4, pad=20)
        
        return vmin, vmax
    
    # Create correlation diagram
    vmin_corr, vmax_corr = create_single_diagram(ax_corr, correlation_matrix,
                                                cmap_corr, "Correlation", -1, 1)
    
    # Create covariance diagram if requested
    if show_both_matrices:
        vmin_cov, vmax_cov = create_single_diagram(ax_cov, covariance_matrix,
                                                  cmap_cov, "Covariance")
    
    # Add statistics panel
    ax_stats.axis('off')
    
    # Calculate statistics
    stats_text = [
        "📊 Dataset Statistics",
        f"Rows: {len(df_selected)}",
        f"Features: {n_features}",
        "",
        "📈 Feature Means:"
    ]
    
    # Add top and bottom features by mean
    sorted_indices = np.argsort(feature_means)[::-1]
    stats_text.append("Top 5 Features:")
    for idx in sorted_indices[:5]:
        stats_text.append(f"  {NODE_NAMES[idx]}: {feature_means[idx]:.3f}")
    
    stats_text.append("")
    stats_text.append("Bottom 5 Features:")
    for idx in sorted_indices[-5:]:
        stats_text.append(f"  {NODE_NAMES[idx]}: {feature_means[idx]:.3f}")
    
    stats_text.append("")
    stats_text.append("🔗 Strongest Correlations:")
    
    # Find strongest correlations
    corr_flat = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr_value = correlation_matrix[i, j]
            if abs(corr_value) > 0.7:
                corr_flat.append((i, j, corr_value))
    
    corr_flat.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for i, j, corr in corr_flat[:3]:
        stats_text.append(f"  {NODE_NAMES[i]} ↔ {NODE_NAMES[j]}: {corr:.3f}")
    
    # Display statistics
    stats_y = 0.95
    for line in stats_text:
        ax_stats.text(0.05, stats_y, line, fontsize=label_size - 1,
                     verticalalignment='top', transform=ax_stats.transAxes)
        if line.endswith(":") or "📊" in line or "📈" in line or "🔗" in line:
            stats_y -= 0.05
        else:
            stats_y -= 0.04
    
    plt.suptitle(f"Multi-Layer Feature Analysis ({len(df_selected)} rows)",
                fontsize=label_size + 8, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig

# ==================== RADIAL HEATMAP FUNCTION ====================
def create_radial_heatmap(df_selected, colormap_name='viridis',
                         metric='correlation', label_size=12,
                         figsize=(14, 12)):
    """
    Create a radial heatmap visualization of correlation/covariance matrix
    
    Parameters:
    -----------
    df_selected : DataFrame
        DataFrame containing selected rows
    colormap_name : str
        Colormap for heatmap
    metric : str
        'correlation' or 'covariance'
    label_size : int
        Font size for labels
    figsize : tuple
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib.figure.Figure
        The radial heatmap figure
    """
    # Calculate matrix
    if metric == 'correlation':
        matrix = df_selected[FEATURE_COLS].corr().fillna(0).values
        metric_label = "Correlation"
        vmin, vmax = -1, 1
    else:
        matrix = df_selected[FEATURE_COLS].cov().fillna(0).values
        metric_label = "Covariance"
        max_abs = np.max(np.abs(matrix))
        vmin, vmax = -max_abs, max_abs
    
    n_features = len(FEATURE_COLS)
    
    # Create figure
    fig, (ax_radial, ax_matrix, ax_stats) = plt.subplots(1, 3, figsize=figsize,
                                                         gridspec_kw={'width_ratios': [2, 2, 1]})
    
    # Setup radial plot
    ax_radial = plt.subplot(131, projection='polar')
    ax_radial.set_theta_zero_location("N")
    ax_radial.set_theta_direction(-1)
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
    angles = np.roll(angles, -1)
    
    # Create colormap
    cmap = cm.get_cmap(colormap_name)
    
    # Create radial grid
    radii = np.linspace(0.5, 2, n_features)
    
    # Plot radial heatmap
    for i in range(n_features):
        for j in range(n_features):
            if i == j:
                continue
            
            value = matrix[i, j]
            abs_value = abs(value)
            
            # Skip very small values
            if abs_value < 0.2:
                continue
            
            # Calculate radial position
            theta = (angles[i] + angles[j]) / 2
            radius = (radii[i] + radii[j]) / 2
            
            # Calculate color
            color = cmap((value - vmin) / (vmax - vmin))
            
            # Plot as a wedge or circle
            wedge_size = abs_value * 0.3
            wedge = Wedge((0, 0), radius + wedge_size/2,
                         np.degrees(theta - wedge_size/2),
                         np.degrees(theta + wedge_size/2),
                         width=wedge_size,
                         facecolor=color,
                         edgecolor='white',
                         linewidth=0.5,
                         alpha=0.7)
            ax_radial.add_patch(wedge)
    
    # Add feature labels
    for idx, angle in enumerate(angles):
        ax_radial.text(angle, 2.5, NODE_NAMES[idx],
                      ha='center', va='center',
                      fontsize=label_size, fontweight='bold',
                      rotation=np.degrees(angle),
                      rotation_mode='anchor')
    
    ax_radial.set_ylim(0, 2.8)
    ax_radial.grid(True, alpha=0.3)
    ax_radial.set_title(f"Radial {metric_label} Map", fontsize=label_size + 4, pad=20)
    
    # Create traditional matrix heatmap
    im = ax_matrix.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
    
    # Add feature labels
    ax_matrix.set_xticks(range(n_features))
    ax_matrix.set_yticks(range(n_features))
    ax_matrix.set_xticklabels(NODE_NAMES, rotation=45, ha='right', fontsize=label_size - 2)
    ax_matrix.set_yticklabels(NODE_NAMES, fontsize=label_size - 2)
    
    # Add values to matrix
    for i in range(n_features):
        for j in range(n_features):
            value = matrix[i, j]
            if abs(value) > 0.3:  # Only show significant values
                color = 'white' if abs(value) > 0.7 else 'black'
                ax_matrix.text(j, i, f'{value:.2f}',
                              ha='center', va='center',
                              color=color, fontsize=label_size - 4)
    
    ax_matrix.set_title(f"{metric_label} Matrix", fontsize=label_size + 4, pad=20)
    
    # Add colorbar for matrix
    cbar = plt.colorbar(im, ax=ax_matrix, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label, fontsize=label_size)
    cbar.ax.tick_params(labelsize=label_size - 2)
    
    # Add statistics panel
    ax_stats.axis('off')
    
    # Calculate statistics
    stats_text = [
        f"{metric_label} Statistics",
        f"Features: {n_features}",
        f"Rows: {len(df_selected)}",
        "",
        "Matrix Statistics:",
        f"Min: {matrix.min():.3f}",
        f"Max: {matrix.max():.3f}",
        f"Mean: {matrix.mean():.3f}",
        f"Std: {matrix.std():.3f}",
        "",
        "Strong Relationships:"
    ]
    
    # Find strongest relationships
    strong_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            value = matrix[i, j]
            if abs(value) > 0.8:
                strong_pairs.append((i, j, value))
    
    strong_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    for i, j, value in strong_pairs[:5]:
        stats_text.append(f"{NODE_NAMES[i]}-{NODE_NAMES[j]}: {value:.3f}")
    
    # Display statistics
    stats_y = 0.95
    for line in stats_text:
        ax_stats.text(0.05, stats_y, line, fontsize=label_size - 1,
                     verticalalignment='top', transform=ax_stats.transAxes)
        stats_y -= 0.05
    
    plt.suptitle(f"{metric_label} Analysis: {len(df_selected)} Rows",
                fontsize=label_size + 8, fontweight='bold', y=0.98)
    plt.tight_layout()
    return fig

# ==================== TEMPORAL DYNAMICS ANALYSIS (NEW FEATURE) ====================
def create_temporal_chord_animation(df, time_col='step', window_size=5,
                                   colormap_name='RdBu', figsize=(14, 14)):
    """
    Create multiple chord diagrams showing temporal evolution of relationships
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame with time-series data
    time_col : str
        Column name containing time/step information
    window_size : int
        Number of consecutive rows to include in each window
    colormap_name : str
        Colormap for edges
    figsize : tuple
        Figure size
    
    Returns:
    --------
    list of Figures
        List of chord diagram figures for each time window
    """
    if time_col not in df.columns:
        time_col = df.index.name if df.index.name else None
    
    if time_col is None:
        # Use row index as time
        time_values = df.index.tolist()
    else:
        time_values = df[time_col].tolist()
    
    n_windows = max(1, len(df) - window_size + 1)
    figures = []
    
    for i in range(0, n_windows, max(1, n_windows // 10)):  # Limit to ~10 windows
        window_df = df.iloc[i:i+window_size]
        time_start = time_values[i] if i < len(time_values) else i
        time_end = time_values[min(i+window_size-1, len(time_values)-1)] if i+window_size-1 < len(time_values) else i+window_size-1
        
        fig = create_correlation_chord_diagram(
            df_selected=window_df,
            colormap_name=colormap_name,
            metric='correlation',
            label_size=10,
            node_size=80,
            edge_width_scale=2,
            edge_threshold=0.3,
            show_node_values=False,
            show_edge_values=False,
            normalize_nodes=True,
            figsize=figsize,
            title_prefix=f"Time Window {time_start}-{time_end} ({window_size} rows) - "
        )
        figures.append(fig)
    
    return figures

# ==================== STATISTICAL SIGNIFICANCE OVERLAY (NEW FEATURE) ====================
def add_significance_indicators(ax, df, feature_pairs, alpha=0.05):
    """
    Add statistical significance indicators to connections
    
    Parameters:
    -----------
    ax : matplotlib axis
        The axis to add indicators to
    df : DataFrame
        The data DataFrame
    feature_pairs : list of tuples
        List of (feature1, feature2) pairs to test
    alpha : float
        Significance level
    
    Returns:
    --------
    dict
        Dictionary mapping feature pairs to significance status
    """
    significance_results = {}
    
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            # Perform correlation test
            corr, p_value = stats.pearsonr(df[feat1].dropna(), df[feat2].dropna())
            is_significant = p_value < alpha
            significance_results[(feat1, feat2)] = {
                'correlation': corr,
                'p_value': p_value,
                'significant': is_significant
            }
    
    return significance_results

# ==================== MAIN FUNCTION ====================
def main():
    st.set_page_config(
        page_title="Advanced Correlation Chord Diagrams",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📊 Advanced Correlation & Covariance Chord Diagrams")
    st.markdown("""
    Visualize statistical relationships between features using comprehensive chord diagrams.
    Features are placed on the circumference, with connections showing correlation/covariance strength.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Data Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type="csv",
            help="Upload a CSV file containing the 15 required features"
        )
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Upload File", "Use Example Data"],
            index=1 if uploaded_file is None else 0
        )
        
        # Load data
        if data_source == "Upload File" and uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df)} rows from uploaded file")
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {e}")
                st.stop()
        else:
            df = pd.read_csv(io.StringIO(EXAMPLE_CSV))
            st.info("ℹ️ Using built-in example dataset (31 rows)")
        
        # Validate columns
        missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
            st.stop()
        
        # Visualization selection
        st.header("🎨 Visualization Type")
        viz_type = st.selectbox(
            "Select Visualization",
            ["Correlation Chord Diagram",
             "Multi-Layer Analysis",
             "Radial Heatmap",
             "Comparative Analysis",
             "Temporal Dynamics"],  # NEW OPTION
            help="Choose the type of visualization to generate"
        )
        
        # Row selection
        st.header("📋 Data Selection")
        all_rows = df.index.tolist()
        
        if viz_type in ["Comparative Analysis", "Temporal Dynamics"]:
            # For comparative/temporal analysis, allow selection of multiple row ranges
            row_ranges = st.slider(
                "Select Row Ranges to Compare",
                min_value=0,
                max_value=len(df)-1,
                value=(0, min(10, len(df)-1)),
                help="Select range of rows for analysis"
            )
            selected_rows = list(range(row_ranges[0], row_ranges[1] + 1))
        else:
            # For single visualizations, select all or specific rows
            use_all_rows = st.checkbox("Use All Rows", value=True)
            if use_all_rows:
                selected_rows = all_rows
            else:
                selected_rows = st.multiselect(
                    "Select Specific Rows",
                    options=all_rows,
                    default=all_rows[:10] if len(all_rows) >= 10 else all_rows,
                    help="Select specific rows to include in analysis"
                )
        
        if not selected_rows:
            st.warning("⚠️ Please select at least one row to visualize")
            st.stop()
        
        st.info(f"📋 Selected {len(selected_rows)} rows for analysis")
        
        # Visualization settings
        st.header("🎨 Visualization Settings")
        
        if viz_type == "Correlation Chord Diagram":
            st.subheader("Connection Settings")
            metric = st.radio(
                "Connection Metric",
                ["correlation", "covariance"],
                help="Choose whether to show correlation or covariance"
            )
            edge_threshold = st.slider(
                "Connection Threshold",
                0.0, 1.0, 0.3,
                help="Minimum absolute value to show connection"
            )
            edge_width_scale = st.slider(
                "Connection Width Scale",
                1.0, 10.0, 3.0,
                help="Scale factor for connection widths"
            )
            show_edge_values = st.checkbox("Show Connection Values", value=False)
            show_node_values = st.checkbox("Show Node Values (Means)", value=True)
            
            # Colormap selection for edges
            st.subheader("Color Scheme")
            edge_cmap = st.selectbox(
                "Connection Colormap",
                STAT_COLORMAPS,
                index=STAT_COLORMAPS.index('RdBu'),
                help="Colormap for correlation/covariance values"
            )
            
            # Preview colormap
            try:
                edge_cmap_preview = plt.get_cmap(edge_cmap)
                colors = [edge_cmap_preview(i) for i in np.linspace(0, 1, 10)]
                st.write("**Connection Colormap Preview:**")
                st.markdown(
                    '<div style="display: flex; height: 20px; border-radius: 2px; overflow: hidden; margin-bottom: 10px;">' +
                    ''.join([f'<div style="flex:1; background-color: rgba({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)},{c[3]});"></div>'
                            for c in colors]) +
                    '</div>',
                    unsafe_allow_html=True
                )
            except:
                pass
        
        elif viz_type == "Multi-Layer Analysis":
            st.subheader("Multi-Layer Settings")
            show_both = st.checkbox("Show Both Correlation & Covariance", value=True)
            col1, col2 = st.columns(2)
            with col1:
                corr_cmap = st.selectbox(
                    "Correlation Colormap",
                    STAT_COLORMAPS,
                    index=STAT_COLORMAPS.index('RdBu')
                )
            with col2:
                cov_cmap = st.selectbox(
                    "Covariance Colormap",
                    STAT_COLORMAPS,
                    index=STAT_COLORMAPS.index('coolwarm')
                )
        
        elif viz_type == "Radial Heatmap":
            st.subheader("Heatmap Settings")
            metric = st.radio(
                "Matrix Type",
                ["correlation", "covariance"]
            )
            heatmap_cmap = st.selectbox(
                "Heatmap Colormap",
                ALL_COLORMAPS,
                index=ALL_COLORMAPS.index('viridis')
            )
        
        elif viz_type == "Temporal Dynamics":
            st.subheader("Temporal Settings")
            window_size = st.slider(
                "Window Size (rows)",
                3, min(15, len(df)), 5,
                help="Number of consecutive rows in each time window"
            )
            temporal_cmap = st.selectbox(
                "Connection Colormap",
                STAT_COLORMAPS,
                index=STAT_COLORMAPS.index('RdBu')
            )
        
        # Common settings
        st.subheader("Common Settings")
        label_size = st.slider("Label Font Size", 8, 24, 12)
        col1, col2 = st.columns(2)
        with col1:
            fig_width = st.slider("Figure Width", 10, 24, 16)
        with col2:
            fig_height = st.slider("Figure Height", 8, 20, 12)
        
        # Export settings
        st.header("💾 Export")
        export_dpi = st.slider("Export DPI", 100, 300, 150)
        export_format = st.selectbox("Export Format", ["PNG", "PDF", "SVG"])
        
        # Analysis settings
        st.header("🔬 Analysis")
        compute_stats = st.checkbox("Compute Detailed Statistics", value=True)
        show_significance = st.checkbox("Show Statistical Significance", value=False)
    
    # Main content area
    st.header(f"🎨 {viz_type}")
    df_selected = df.loc[selected_rows]
    
    # Display dataset summary
    with st.expander("📋 Dataset Summary", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df_selected))
            st.metric("Features", len(FEATURE_COLS))
        with col2:
            means = df_selected[FEATURE_COLS].mean()
            max_feature = means.idxmax()
            min_feature = means.idxmin()
            st.metric("Highest Mean", f"{max_feature}: {means[max_feature]:.3f}")
            st.metric("Lowest Mean", f"{min_feature}: {means[min_feature]:.3f}")
        with col3:
            stds = df_selected[FEATURE_COLS].std()
            max_std_feature = stds.idxmax()
            min_std_feature = stds.idxmin()
            st.metric("Highest Variance", f"{max_std_feature}: {stds[max_std_feature]:.3f}")
            st.metric("Lowest Variance", f"{min_std_feature}: {stds[min_std_feature]:.3f}")
    
    # Generate visualization based on selected type
    with st.spinner("🎨 Generating visualization..."):
        try:
            fig = None
            figs_list = []  # For comparative/temporal analysis
            
            if viz_type == "Correlation Chord Diagram":
                fig = create_correlation_chord_diagram(
                    df_selected=df_selected,
                    colormap_name=edge_cmap,
                    metric=metric,
                    label_size=label_size,
                    node_size=100,
                    edge_width_scale=edge_width_scale,
                    edge_threshold=edge_threshold,
                    show_node_values=show_node_values,
                    show_edge_values=show_edge_values,
                    normalize_nodes=True,
                    figsize=(fig_width, fig_height),
                    title_prefix=f"{len(selected_rows)} Rows - "
                )
            
            elif viz_type == "Multi-Layer Analysis":
                fig = create_multi_layer_chord_diagram(
                    df_selected=df_selected,
                    colormap_corr=corr_cmap,
                    colormap_cov=cov_cmap,
                    label_size=label_size,
                    node_size=100,
                    figsize=(fig_width, fig_height),
                    show_both_matrices=show_both
                )
            
            elif viz_type == "Radial Heatmap":
                fig = create_radial_heatmap(
                    df_selected=df_selected,
                    colormap_name=heatmap_cmap,
                    metric=metric,
                    label_size=label_size,
                    figsize=(fig_width, fig_height)
                )
            
            elif viz_type == "Temporal Dynamics":
                # Create temporal chord diagrams
                temporal_figs = create_temporal_chord_animation(
                    df=df.loc[selected_rows],
                    time_col='step' if 'step' in df.columns else None,
                    window_size=window_size,
                    colormap_name=temporal_cmap,
                    figsize=(fig_width, fig_height)
                )
                
                # Display in columns or expanders
                st.subheader("Temporal Evolution of Relationships")
                for idx, temp_fig in enumerate(temporal_figs):
                    with st.expander(f"Time Window {idx + 1}/{len(temporal_figs)}", expanded=(idx == 0)):
                        st.pyplot(temp_fig, use_container_width=True)
                        
                        # Individual download for each temporal window
                        buf = io.BytesIO()
                        temp_fig.savefig(buf, format='png', dpi=export_dpi, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            f"💾 Download Window {idx + 1}",
                            buf,
                            f"temporal_window_{idx + 1}.png",
                            "image/png",
                            key=f"temporal_download_{idx}"
                        )
                    plt.close(temp_fig)
                
                # Don't show single figure
                fig = None
            
            else:  # Comparative Analysis
                # Create multiple diagrams for comparison
                if len(selected_rows) > 20:
                    # Split into groups for comparison
                    n_groups = min(3, max(2, len(selected_rows) // 10))
                    groups = np.array_split(selected_rows, n_groups)
                else:
                    groups = [selected_rows]
                
                for i, group in enumerate(groups):
                    if len(group) == 0:
                        continue
                    
                    df_group = df.loc[group]
                    group_fig = create_correlation_chord_diagram(
                        df_selected=df_group,
                        colormap_name='RdBu',
                        metric='correlation',
                        label_size=label_size - 2,
                        node_size=80,
                        edge_width_scale=2,
                        edge_threshold=0.3,
                        show_node_values=True,
                        show_edge_values=False,
                        normalize_nodes=True,
                        figsize=(fig_width/len(groups), fig_height),
                        title_prefix=f"Group {i+1} ({len(group)} rows) - "
                    )
                    figs_list.append((group_fig, f"group_{i+1}"))
                
                # Display in columns
                st.subheader("Comparative Analysis")
                cols = st.columns(len(figs_list))
                for idx, (col, (fig_item, name)) in enumerate(zip(cols, figs_list)):
                    with col:
                        st.pyplot(fig_item, use_container_width=True)
                        
                        # Group-specific download
                        buf = io.BytesIO()
                        fig_item.savefig(buf, format='png', dpi=export_dpi, bbox_inches='tight')
                        buf.seek(0)
                        st.download_button(
                            f"💾 Download {name}",
                            buf,
                            f"{name}_diagram.png",
                            "image/png",
                            key=f"comparative_download_{idx}"
                        )
                    plt.close(fig_item)
                
                # Don't show single figure for comparative analysis
                fig = None
            
            # Display single figure for non-comparative analyses
            if fig is not None:
                st.pyplot(fig, use_container_width=True)
                
                # Download button
                buf = io.BytesIO()
                if export_format == "PNG":
                    fig.savefig(buf, format='png', dpi=export_dpi, bbox_inches='tight')
                    mime_type = "image/png"
                    file_ext = "png"
                elif export_format == "PDF":
                    fig.savefig(buf, format='pdf', bbox_inches='tight')
                    mime_type = "application/pdf"
                    file_ext = "pdf"
                else:  # SVG
                    fig.savefig(buf, format='svg', bbox_inches='tight')
                    mime_type = "image/svg+xml"
                    file_ext = "svg"
                
                buf.seek(0)
                viz_name = viz_type.lower().replace(" ", "_")
                st.download_button(
                    label=f"💾 Download {viz_type} ({export_format})",
                    data=buf,
                    file_name=f"{viz_name}_diagram.{file_ext}",
                    mime=mime_type,
                    key=f"download_{viz_name}"
                )
                plt.close(fig)
            
            # Display detailed statistics if requested
            if compute_stats and fig is not None:
                st.header("📊 Detailed Statistics")
                
                # Calculate correlation matrix
                corr_matrix = df_selected[FEATURE_COLS].corr()
                cov_matrix = df_selected[FEATURE_COLS].cov()
                
                # Display in tabs
                tab1, tab2, tab3 = st.tabs(["📈 Correlation Matrix", "📉 Covariance Matrix", "🔍 Feature Statistics"])
                
                with tab1:
                    # Style correlation matrix
                    def color_corr(val):
                        if val < -0.7:
                            color = 'darkred'
                        elif val < -0.3:
                            color = 'red'
                        elif val < 0.3:
                            color = 'lightgray'
                        elif val < 0.7:
                            color = 'lightgreen'
                        else:
                            color = 'darkgreen'
                        return f'background-color: {color}; color: white;'
                    
                    st.dataframe(corr_matrix.style.format("{:.3f}").applymap(color_corr),
                                use_container_width=True)
                    
                    # Strongest correlations
                    st.subheader("🔗 Strongest Correlations")
                    corr_pairs = []
                    for i in range(len(FEATURE_COLS)):
                        for j in range(i + 1, len(FEATURE_COLS)):
                            value = corr_matrix.iloc[i, j]
                            corr_pairs.append((FEATURE_COLS[i], FEATURE_COLS[j], value))
                    
                    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                    cols = st.columns(3)
                    for idx, (feature1, feature2, value) in enumerate(corr_pairs[:9]):
                        with cols[idx % 3]:
                            emoji = "🟢" if value > 0 else "🔴"
                            st.metric(f"{feature1} ↔ {feature2}", f"{value:.3f}")
                
                with tab2:
                    st.dataframe(cov_matrix.style.format("{:.3f}"), use_container_width=True)
                    
                    # Covariance summary
                    st.subheader("📊 Covariance Summary")
                    cov_stats = pd.DataFrame({
                        'Mean': cov_matrix.mean(),
                        'Std': cov_matrix.std(),
                        'Min': cov_matrix.min(),
                        'Max': cov_matrix.max()
                    })
                    st.dataframe(cov_stats.style.format("{:.3f}"), use_container_width=True)
                
                with tab3:
                    feature_stats = df_selected[FEATURE_COLS].agg(['mean', 'std', 'min', 'max', 'median']).T
                    feature_stats['cv'] = feature_stats['std'] / feature_stats['mean']  # Coefficient of variation
                    feature_stats = feature_stats.round(4)
                    st.dataframe(feature_stats, use_container_width=True)
                    
                    # Distribution plots for top 3 features
                    st.subheader("📈 Feature Distributions")
                    top_features = feature_stats.sort_values('mean', ascending=False).index[:3]
                    cols = st.columns(3)
                    for idx, feature in enumerate(top_features):
                        with cols[idx]:
                            fig_dist, ax = plt.subplots(figsize=(4, 3))
                            ax.hist(df_selected[feature], bins=15, alpha=0.7, color=plt.cm.Set1(idx))
                            ax.set_title(f"{feature} Distribution")
                            ax.set_xlabel("Value")
                            ax.set_ylabel("Frequency")
                            st.pyplot(fig_dist, use_container_width=True)
                            plt.close(fig_dist)
            
            # Show statistical significance if requested
            if show_significance and fig is not None:
                st.header("🔬 Statistical Significance")
                
                # Find all feature pairs with strong correlations
                corr_matrix = df_selected[FEATURE_COLS].corr()
                strong_pairs = []
                for i in range(len(FEATURE_COLS)):
                    for j in range(i + 1, len(FEATURE_COLS)):
                        value = corr_matrix.iloc[i, j]
                        if abs(value) > 0.5:
                            strong_pairs.append((FEATURE_COLS[i], FEATURE_COLS[j]))
                
                # Test significance
                significance_results = add_significance_indicators(None, df_selected, strong_pairs, alpha=0.05)
                
                # Display results
                sig_df = pd.DataFrame([
                    {
                        'Feature 1': pair[0],
                        'Feature 2': pair[1],
                        'Correlation': results['correlation'],
                        'p-value': results['p_value'],
                        'Significant (p<0.05)': '✅' if results['significant'] else '❌'
                    }
                    for pair, results in significance_results.items()
                ])
                
                st.dataframe(sig_df.sort_values('p-value').reset_index(drop=True), use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error generating visualization: {str(e)}")
            with st.expander("Show Full Error Traceback"):
                st.code(traceback.format_exc())
    
    # Footer with explanation
    st.markdown("---")
    with st.expander("📚 How to Interpret These Visualizations"):
        st.markdown("""
        ### Understanding Chord Diagrams with Statistical Connections
        
        **🔵 Node Representation (Features on Circumference):**
        - **Size**: Represents the mean value of each feature across selected rows
        - **Color**: Each feature has a distinct color for identification
        - **Position**: Evenly spaced around the circle
        
        **🔗 Connection Representation (Statistical Relationships):**
        - **Color**: Shows strength and direction of relationship
        - **Red/Blue (RdBu)**: Red = positive correlation, Blue = negative correlation
        - **Intensity**: Darker colors = stronger relationships
        - **Width**: Thicker lines = stronger absolute correlation/covariance
        - **Opacity**: More transparent lines = weaker relationships
        
        **📊 Statistical Metrics:**
        - **Correlation**: Measures linear relationship between features (-1 to 1)
          - +1: Perfect positive relationship
          - 0: No linear relationship
          - -1: Perfect negative relationship
        - **Covariance**: Measures how two features vary together
          - Positive: Features tend to increase together
          - Negative: One increases while the other decreases
        
        **🎯 Interpretation Guidelines:**
        1. Look for **clusters** of strongly connected features
        2. Identify **hub features** with many strong connections
        3. Note **negative correlations** (blue connections)
        4. Observe **isolated features** with few connections
        
        **💡 Tips for Analysis:**
        - Use threshold sliders to reduce clutter
        - Compare different row subsets to see changing relationships
        - Export high-resolution images for detailed inspection
        - Use the statistical tables for precise numerical analysis
        - Check statistical significance to avoid spurious correlations
        """)

if __name__ == "__main__":
    main()
