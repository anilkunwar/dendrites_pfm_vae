import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI issues
import matplotlib.cm as cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle, Wedge, FancyBboxPatch
import matplotlib.colors as mcolors
import io
import warnings
import base64
from typing import List, Dict, Tuple, Optional
import gc

warnings.filterwarnings('ignore')

# Set global matplotlib parameters to reduce memory usage
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['agg.path.chunksize'] = 10000

# Hardcoded example data
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

# Generate extensive colormap list
ALL_COLORMAPS = sorted(set([
    'viridis', 'plasma', 'inferno', 'magma', 'cividis',
    'coolwarm', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
    'turbo', 'jet', 'rainbow', 'hsv', 'spring', 'summer', 
    'autumn', 'winter', 'hot', 'cool', 'copper', 'bone',
    'pink', 'gray', 'Greys', 'Purples', 'Blues', 'Greens', 
    'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
    'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
    'BuGn', 'YlGn'
]))

# Statistical colormaps
STAT_COLORMAPS = ['coolwarm', 'RdBu', 'RdGy', 'PiYG', 'PRGn', 
                  'RdYlBu', 'RdYlGn', 'Spectral', 'bwr', 'seismic']

FEATURE_COLS = ['t','POT_LEFT','fo','Al','Bl','Cl','As','Bs','Cs','cleq','cseq','L1o','L2o','ko','Noise']
NODE_NAMES = FEATURE_COLS

@st.cache_data(ttl=3600, max_entries=10)
def load_data(uploaded_file=None, use_example=True):
    """Load data with caching to improve performance"""
    if not use_example and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None
    else:
        df = pd.read_csv(io.StringIO(EXAMPLE_CSV))
        return df

def compute_statistics(df_selected):
    """Compute all required statistics efficiently"""
    stats = {}
    
    # Basic statistics
    stats['feature_means'] = df_selected[FEATURE_COLS].mean().values
    stats['feature_stds'] = df_selected[FEATURE_COLS].std().values
    stats['feature_mins'] = df_selected[FEATURE_COLS].min().values
    stats['feature_maxs'] = df_selected[FEATURE_COLS].max().values
    
    # Correlation and covariance matrices
    stats['correlation_matrix'] = df_selected[FEATURE_COLS].corr().fillna(0).values
    stats['covariance_matrix'] = df_selected[FEATURE_COLS].cov().fillna(0).values
    
    return stats

def create_simple_chord_diagram(stats, metric='correlation', colormap_name='coolwarm',
                               label_size=12, edge_threshold=0.3, max_connections=50,
                               figsize=(12, 12), title=""):
    """
    Create an optimized chord diagram with reduced complexity
    """
    # Use correlation or covariance matrix
    if metric == 'correlation':
        matrix = stats['correlation_matrix']
        vmin, vmax = -1, 1
        metric_label = "Correlation"
    else:
        matrix = stats['covariance_matrix']
        max_abs = np.max(np.abs(matrix))
        vmin, vmax = -max_abs, max_abs
        metric_label = "Covariance"
    
    n_features = len(FEATURE_COLS)
    feature_means = stats['feature_means']
    
    # Setup figure with constrained layout
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 2.5)
    ax.axis('off')
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
    angles = np.roll(angles, -1)
    
    # Get colormaps
    edge_cmap = cm.get_cmap(colormap_name)
    node_cmap = cm.get_cmap('tab20c', n_features)
    
    # Calculate node sizes (normalized)
    if feature_means.max() > 0:
        node_sizes = 0.5 + (feature_means / feature_means.max()) * 0.5
    else:
        node_sizes = np.ones(n_features) * 0.75
    
    # Draw nodes
    for idx, (angle, size) in enumerate(zip(angles, node_sizes)):
        # Draw node as a circle
        node_color = node_cmap(idx)
        circle = Circle((0, 0), 1.0 + size/2, 
                       facecolor=node_color,
                       edgecolor='white',
                       linewidth=2,
                       alpha=0.9,
                       transform=ax.transData._b + 
                       matplotlib.transforms.Affine2D().rotate(angle))
        ax.add_patch(circle)
        
        # Add label
        label_angle = angle
        label_radius = 1.0 + size + 0.3
        
        rotation = np.degrees(label_angle)
        if 90 <= rotation <= 270:
            rotation += 180
            align = "right"
        else:
            align = "left"
            
        ax.text(label_angle, label_radius, NODE_NAMES[idx],
               ha=align, va='center',
               fontsize=label_size,
               fontweight='bold',
               rotation=rotation,
               rotation_mode='anchor',
               bbox=dict(boxstyle="round,pad=0.2", 
                        facecolor='white', 
                        alpha=0.8,
                        edgecolor='lightgray'))
    
    # Collect connections (limit to strongest ones)
    connections = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            value = matrix[i, j]
            abs_value = abs(value)
            if abs_value > edge_threshold:
                connections.append((i, j, value, abs_value))
    
    # Sort by absolute value and limit number
    connections.sort(key=lambda x: x[3], reverse=True)
    if len(connections) > max_connections:
        connections = connections[:max_connections]
    
    # Draw connections with varying opacity and width
    for i, j, value, abs_value in connections:
        # Calculate connection properties
        theta1, theta2 = angles[i], angles[j]
        radius1 = 1.0 + node_sizes[i]/2
        radius2 = 1.0 + node_sizes[j]/2
        
        # Control point for Bezier curve
        control_radius = (radius1 + radius2) / 2 * 0.8
        control_theta = (theta1 + theta2) / 2
        
        # Color based on value
        color = edge_cmap((value - vmin) / (vmax - vmin))
        
        # Width based on absolute value
        width = abs_value * 2
        
        # Create simplified path (just the curve, not filled)
        t = np.linspace(0, 1, 20)
        curve_theta = theta1 + (theta2 - theta1) * t
        curve_radius = radius1 + (radius2 - radius1) * t
        
        # Add sinusoidal modulation for curve shape
        curve_mod = np.sin(np.pi * t) * 0.3
        curve_radius = curve_radius + curve_mod
        
        # Plot as line instead of filled patch
        ax.plot(curve_theta, curve_radius, 
               color=color,
               linewidth=width,
               alpha=0.7,
               solid_capstyle='round')
    
    # Add title
    ax.set_title(f"{title}\n{metric_label} Chord Diagram", 
                fontsize=label_size + 4, 
                fontweight='bold',
                pad=30)
    
    # Add colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=edge_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, aspect=50)
    cbar.set_label(f'{metric_label} Value', fontsize=label_size)
    cbar.ax.tick_params(labelsize=label_size - 2)
    
    plt.tight_layout()
    return fig

def create_matrix_heatmap(stats, metric='correlation', colormap_name='coolwarm',
                         figsize=(10, 8), title=""):
    """Create a matrix heatmap visualization"""
    if metric == 'correlation':
        matrix = stats['correlation_matrix']
        vmin, vmax = -1, 1
        metric_label = "Correlation"
    else:
        matrix = stats['covariance_matrix']
        max_abs = np.max(np.abs(matrix))
        vmin, vmax = -max_abs, max_abs
        metric_label = "Covariance"
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(matrix, cmap=colormap_name, vmin=vmin, vmax=vmax,
                   aspect='auto', interpolation='nearest')
    
    # Add labels
    ax.set_xticks(range(len(FEATURE_COLS)))
    ax.set_yticks(range(len(FEATURE_COLS)))
    ax.set_xticklabels(FEATURE_COLS, rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(FEATURE_COLS, fontsize=10)
    
    # Add grid
    ax.set_xticks(np.arange(-.5, len(FEATURE_COLS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(FEATURE_COLS), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Add values for significant correlations
    threshold = 0.5 if metric == 'correlation' else np.percentile(np.abs(matrix), 75)
    for i in range(len(FEATURE_COLS)):
        for j in range(len(FEATURE_COLS)):
            value = matrix[i, j]
            if abs(value) > threshold:
                color = 'white' if abs(value) > 0.7 else 'black'
                ax.text(j, i, f'{value:.2f}', 
                       ha='center', va='center', 
                       color=color, fontsize=9, fontweight='bold')
    
    ax.set_title(f"{title}\n{metric_label} Matrix", fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label, fontsize=12)
    
    plt.tight_layout()
    return fig

def create_radial_plot(stats, metric='correlation', colormap_name='coolwarm',
                      figsize=(10, 10), title=""):
    """Create a radial plot visualization"""
    if metric == 'correlation':
        matrix = stats['correlation_matrix']
        vmin, vmax = -1, 1
        metric_label = "Correlation"
    else:
        matrix = stats['covariance_matrix']
        max_abs = np.max(np.abs(matrix))
        vmin, vmax = -max_abs, max_abs
        metric_label = "Covariance"
    
    n_features = len(FEATURE_COLS)
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False)
    
    # Get colormap
    cmap = cm.get_cmap(colormap_name)
    
    # Create radial grid
    radii = np.linspace(0.5, 2.0, n_features)
    
    # Plot connections as radial lines
    for i in range(n_features):
        for j in range(i + 1, n_features):
            value = matrix[i, j]
            abs_value = abs(value)
            
            if abs_value > 0.3:  # Threshold
                # Calculate color
                color = cmap((value - vmin) / (vmax - vmin))
                
                # Plot radial line
                ax.plot([angles[i], angles[j]], [radii[i], radii[j]],
                       color=color,
                       linewidth=abs_value * 3,
                       alpha=0.6,
                       solid_capstyle='round')
    
    # Add feature markers
    for idx, (angle, radius) in enumerate(zip(angles, radii)):
        ax.plot(angle, radius, 'o', markersize=15,
               color=cm.get_cmap('tab20c')(idx),
               markeredgecolor='white',
               markeredgewidth=2)
        
        # Add labels
        label_angle = angle
        label_radius = radius + 0.3
        
        rotation = np.degrees(label_angle)
        if 90 <= rotation <= 270:
            rotation += 180
            align = "right"
        else:
            align = "left"
            
        ax.text(label_angle, label_radius, NODE_NAMES[idx],
               ha=align, va='center',
               fontsize=11,
               fontweight='bold',
               rotation=rotation,
               rotation_mode='anchor')
    
    ax.set_ylim(0, 2.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title}\n{metric_label} Radial Plot", 
                fontsize=14, fontweight='bold', pad=30)
    
    # Add colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, aspect=50)
    cbar.set_label(f'{metric_label} Value', fontsize=12)
    
    plt.tight_layout()
    return fig

def create_statistics_panel(stats, num_rows):
    """Create a statistics summary panel"""
    n_features = len(FEATURE_COLS)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Feature means
    ax1 = axes[0, 0]
    means = stats['feature_means']
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, n_features))
    bars1 = ax1.barh(range(n_features), means, color=colors)
    ax1.set_yticks(range(n_features))
    ax1.set_yticklabels(FEATURE_COLS)
    ax1.set_xlabel('Mean Value')
    ax1.set_title('Feature Means')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Feature standard deviations
    ax2 = axes[0, 1]
    stds = stats['feature_stds']
    colors = cm.get_cmap('plasma')(np.linspace(0, 1, n_features))
    bars2 = ax2.barh(range(n_features), stds, color=colors)
    ax2.set_yticks(range(n_features))
    ax2.set_yticklabels(FEATURE_COLS)
    ax2.set_xlabel('Standard Deviation')
    ax2.set_title('Feature Variability')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Plot 3: Correlation distribution
    ax3 = axes[1, 0]
    corr_flat = stats['correlation_matrix'][np.triu_indices(n_features, k=1)]
    ax3.hist(corr_flat, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Correlation Coefficient')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Correlations')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Top correlations
    ax4 = axes[1, 1]
    # Find top correlations
    corr_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            value = stats['correlation_matrix'][i, j]
            corr_pairs.append((i, j, abs(value)))
    
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    top_n = min(10, len(corr_pairs))
    
    top_labels = []
    top_values = []
    for i, j, val in corr_pairs[:top_n]:
        top_labels.append(f"{FEATURE_COLS[i]}-{FEATURE_COLS[j]}")
        top_values.append(val)
    
    colors = cm.get_cmap('coolwarm')(np.linspace(0, 1, top_n))
    bars4 = ax4.barh(range(top_n), top_values, color=colors)
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels(top_labels, fontsize=9)
    ax4.set_xlabel('Absolute Correlation')
    ax4.set_title(f'Top {top_n} Correlations')
    ax4.grid(True, alpha=0.3, axis='x')
    
    fig.suptitle(f'Statistical Summary ({num_rows} rows)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def main():
    st.set_page_config(
        page_title="Feature Correlation Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Add custom CSS
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
    }
    div[data-baseweb="select"] {
        min-width: 200px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Feature Correlation Explorer")
    st.markdown("""
    Visualize statistical relationships between features using chord diagrams, heatmaps, and radial plots.
    Features are displayed around the circumference with connections showing correlation or covariance.
    """)
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Data Configuration")
        
        # File upload with size limit
        uploaded_file = st.file_uploader(
            "Upload CSV File", 
            type=["csv"],
            help="Maximum file size: 50MB"
        )
        
        # Data source selection
        if uploaded_file is not None:
            if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
                st.error("File too large! Please upload a file smaller than 50MB.")
                use_example = True
            else:
                use_example = False
        else:
            use_example = True
            st.info("Using example dataset")
        
        # Load data
        df = load_data(uploaded_file, use_example)
        
        if df is None:
            st.error("Failed to load data. Please check your file format.")
            st.stop()
        
        # Validate columns
        missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.info(f"Available columns: {', '.join(df.columns.tolist())}")
            st.stop()
        
        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        st.header("📈 Data Selection")
        
        # Row selection with performance warning
        total_rows = len(df)
        if total_rows > 100:
            st.warning(f"Dataset has {total_rows} rows. Selecting many rows may slow down visualization.")
        
        # Simple row selection options
        selection_method = st.radio(
            "Selection Method",
            ["First N rows", "Random sample", "All rows"],
            index=0
        )
        
        if selection_method == "First N rows":
            max_rows = st.slider("Number of rows", 1, min(500, total_rows), min(100, total_rows))
            selected_indices = list(range(min(max_rows, total_rows)))
        elif selection_method == "Random sample":
            sample_size = st.slider("Sample size", 1, min(500, total_rows), min(100, total_rows))
            selected_indices = np.random.choice(total_rows, min(sample_size, total_rows), replace=False).tolist()
        else:  # All rows
            selected_indices = list(range(total_rows))
        
        df_selected = df.iloc[selected_indices].copy()
        
        st.info(f"Selected {len(df_selected)} rows for analysis")
        
        st.header("🎨 Visualization Settings")
        
        # Visualization type
        viz_type = st.selectbox(
            "Visualization Type",
            ["Chord Diagram", "Matrix Heatmap", "Radial Plot", "Statistics Panel"],
            index=0
        )
        
        # Common settings
        colormap_name = st.selectbox(
            "Colormap",
            ALL_COLORMAPS,
            index=ALL_COLORMAPS.index('coolwarm')
        )
        
        label_size = st.slider("Label Size", 8, 16, 12)
        
        if viz_type == "Chord Diagram":
            st.subheader("Chord Diagram Settings")
            metric = st.radio(
                "Connection Metric",
                ["correlation", "covariance"],
                index=0
            )
            
            edge_threshold = st.slider(
                "Minimum Connection Strength",
                0.0, 1.0, 0.3,
                help="Hide connections weaker than this threshold"
            )
            
            max_connections = st.slider(
                "Maximum Connections",
                10, 100, 50,
                help="Limit number of connections to improve performance"
            )
            
            fig_width = st.slider("Figure Width", 8, 16, 12)
            fig_height = st.slider("Figure Height", 8, 16, 12)
        
        elif viz_type == "Matrix Heatmap":
            st.subheader("Heatmap Settings")
            metric = st.radio(
                "Matrix Type",
                ["correlation", "covariance"],
                index=0
            )
            
            fig_width = st.slider("Figure Width", 8, 16, 10)
            fig_height = st.slider("Figure Height", 6, 14, 8)
        
        elif viz_type == "Radial Plot":
            st.subheader("Radial Plot Settings")
            metric = st.radio(
                "Connection Metric",
                ["correlation", "covariance"],
                index=0
            )
            
            fig_width = st.slider("Figure Width", 8, 16, 10)
            fig_height = st.slider("Figure Height", 8, 16, 10)
        
        else:  # Statistics Panel
            fig_width = st.slider("Figure Width", 10, 20, 12)
            fig_height = st.slider("Figure Height", 8, 16, 10)
        
        st.header("⚡ Performance")
        
        # Performance options
        use_caching = st.checkbox("Use Caching", value=True, 
                                 help="Cache computations for faster response")
        
        clear_cache = st.button("Clear Cache")
        if clear_cache:
            st.cache_data.clear()
            st.success("Cache cleared!")
        
        st.header("💾 Export")
        export_dpi = st.slider("Export DPI", 72, 300, 150)
        show_download = st.checkbox("Show Download Options", value=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Statistics", "📋 Data"])
    
    with tab1:
        st.header(f"{viz_type} Visualization")
        
        # Compute statistics
        with st.spinner("Computing statistics..."):
            stats = compute_statistics(df_selected)
        
        # Create visualization
        with st.spinner("Generating visualization..."):
            try:
                if viz_type == "Chord Diagram":
                    fig = create_simple_chord_diagram(
                        stats=stats,
                        metric=metric,
                        colormap_name=colormap_name,
                        label_size=label_size,
                        edge_threshold=edge_threshold,
                        max_connections=max_connections,
                        figsize=(fig_width, fig_height),
                        title=f"Rows {len(df_selected)}"
                    )
                
                elif viz_type == "Matrix Heatmap":
                    fig = create_matrix_heatmap(
                        stats=stats,
                        metric=metric,
                        colormap_name=colormap_name,
                        figsize=(fig_width, fig_height),
                        title=f"Rows {len(df_selected)}"
                    )
                
                elif viz_type == "Radial Plot":
                    fig = create_radial_plot(
                        stats=stats,
                        metric=metric,
                        colormap_name=colormap_name,
                        figsize=(fig_width, fig_height),
                        title=f"Rows {len(df_selected)}"
                    )
                
                else:  # Statistics Panel
                    fig = create_statistics_panel(
                        stats=stats,
                        num_rows=len(df_selected)
                    )
                
                # Display the figure
                st.pyplot(fig, use_container_width=True)
                
                # Download options
                if show_download:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # PNG download
                        buf_png = io.BytesIO()
                        fig.savefig(buf_png, format='png', dpi=export_dpi, 
                                   bbox_inches='tight', facecolor='white')
                        buf_png.seek(0)
                        st.download_button(
                            label="📥 Download PNG",
                            data=buf_png,
                            file_name=f"visualization_{viz_type.replace(' ', '_').lower()}.png",
                            mime="image/png"
                        )
                    
                    with col2:
                        # PDF download
                        buf_pdf = io.BytesIO()
                        fig.savefig(buf_pdf, format='pdf', 
                                   bbox_inches='tight', facecolor='white')
                        buf_pdf.seek(0)
                        st.download_button(
                            label="📥 Download PDF",
                            data=buf_pdf,
                            file_name=f"visualization_{viz_type.replace(' ', '_').lower()}.pdf",
                            mime="application/pdf"
                        )
                    
                    with col3:
                        # SVG download
                        buf_svg = io.BytesIO()
                        fig.savefig(buf_svg, format='svg', 
                                   bbox_inches='tight', facecolor='white')
                        buf_svg.seek(0)
                        st.download_button(
                            label="📥 Download SVG",
                            data=buf_svg,
                            file_name=f"visualization_{viz_type.replace(' ', '_').lower()}.svg",
                            mime="image/svg+xml"
                        )
                
                # Clean up
                plt.close(fig)
                gc.collect()
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    with tab2:
        st.header("Statistical Summary")
        
        # Display correlation matrix
        st.subheader("Correlation Matrix")
        corr_df = df_selected[FEATURE_COLS].corr().round(3)
        
        # Style the correlation matrix
        def color_corr(val):
            color = 'background-color: red' if val < -0.7 else \
                   'background-color: orange' if val < -0.3 else \
                   'background-color: lightgreen' if val < 0.3 else \
                   'background-color: green' if val < 0.7 else \
                   'background-color: darkgreen'
            return color + '; color: white;'
        
        st.dataframe(corr_df.style.applymap(color_corr), use_container_width=True)
        
        # Display covariance matrix
        st.subheader("Covariance Matrix")
        cov_df = df_selected[FEATURE_COLS].cov().round(3)
        st.dataframe(cov_df, use_container_width=True)
        
        # Feature statistics
        st.subheader("Feature Statistics")
        stats_df = pd.DataFrame({
            'Mean': df_selected[FEATURE_COLS].mean(),
            'Std': df_selected[FEATURE_COLS].std(),
            'Min': df_selected[FEATURE_COLS].min(),
            'Max': df_selected[FEATURE_COLS].max(),
            'Median': df_selected[FEATURE_COLS].median()
        }).round(3)
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Strongest correlations
        st.subheader("Strongest Correlations")
        
        # Find top positive and negative correlations
        corr_values = []
        for i in range(len(FEATURE_COLS)):
            for j in range(i + 1, len(FEATURE_COLS)):
                corr = corr_df.iloc[i, j]
                corr_values.append((FEATURE_COLS[i], FEATURE_COLS[j], corr))
        
        # Sort by absolute value
        corr_values.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Display in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Positive Correlations:**")
            pos_corrs = [c for c in corr_values[:10] if c[2] > 0]
            for feature1, feature2, value in pos_corrs[:5]:
                st.metric(f"{feature1} - {feature2}", f"{value:.3f}")
        
        with col2:
            st.write("**Top Negative Correlations:**")
            neg_corrs = [c for c in corr_values if c[2] < 0]
            for feature1, feature2, value in neg_corrs[:5]:
                st.metric(f"{feature1} - {feature2}", f"{value:.3f}", 
                         delta_color="inverse")
    
    with tab3:
        st.header("Data Preview")
        
        # Display selected data
        st.write(f"Showing {len(df_selected)} selected rows:")
        
        # Pagination
        page_size = st.slider("Rows per page", 10, 100, 20)
        total_pages = max(1, len(df_selected) // page_size)
        
        page_num = st.number_input("Page", 1, total_pages, 1)
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(df_selected))
        
        st.dataframe(df_selected.iloc[start_idx:end_idx], use_container_width=True)
        
        # Data summary
        st.subheader("Data Summary")
        summary_stats = {
            'Total Rows': len(df_selected),
            'Total Columns': len(df_selected.columns),
            'Missing Values': df_selected.isnull().sum().sum(),
            'Memory Usage': f"{df_selected.memory_usage(deep=True).sum() / 1024:.1f} KB"
        }
        
        for key, value in summary_stats.items():
            st.write(f"**{key}:** {value}")
        
        # Column information
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df_selected.columns,
            'Type': df_selected.dtypes.astype(str),
            'Non-Null Count': df_selected.notnull().sum(),
            'Unique Values': [df_selected[col].nunique() for col in df_selected.columns]
        })
        
        st.dataframe(col_info, use_container_width=True)
    
    # Footer with explanations
    st.markdown("---")
    with st.expander("ℹ️ About This Visualization"):
        st.markdown("""
        ### How to Interpret the Visualizations
        
        **Chord Diagram:**
        - Features are arranged around the circle
        - Connections show statistical relationships
        - Line thickness = strength of relationship
        - Line color = direction (red = positive, blue = negative)
        
        **Matrix Heatmap:**
        - Traditional correlation/covariance matrix
        - Color intensity = strength of relationship
        - Diagonal = 1.0 (perfect self-correlation)
        
        **Radial Plot:**
        - Features on radial axes
        - Lines connect related features
        - Line properties indicate relationship strength
        
        **Statistical Metrics:**
        - **Correlation**: Measures linear relationship (-1 to 1)
        - **Covariance**: Measures joint variability
        - Values near ±1 indicate strong relationships
        - Values near 0 indicate weak relationships
        
        ### Performance Tips
        1. Limit the number of rows for faster rendering
        2. Increase connection threshold to reduce clutter
        3. Use caching for repeated analyses
        4. Export high-quality images for presentations
        """)

if __name__ == "__main__":
    # Add memory management
    import os
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "50"
    
    # Run the app
    main()
