import nengo
import numpy as np
import itertools
from nengo import spa
from nengo_extras.vision import Gabor, Mask


import os
import png
def load_images(dir, items=None, length=90*14, count=None):
    names = []
    for fn in os.listdir(dir):
        if fn[-4:] == '.png':
            if items is None or fn[:-4] in items:
                names.append(fn[:-4])                      

    if count is not None:
        names = names[:count]

    images = np.empty(shape=(np.size(names), length), dtype='float32')
    for i,fn in enumerate(names):
            r = png.Reader(os.path.join(dir, fn + '.png'))
            r = r.asDirect()
            image_2d = np.vstack(itertools.imap(np.uint8, r[2]))
            image_2d /= 255
            image_1d = image_2d.reshape(1,length)
            images[i] = image_1d
    
    return names, 2*images - 1  # normalize to -1 to 1

def generate_vectors(items, vocab):
    result = np.empty((len(items), vocab.dimensions))
    for i, item in enumerate(items):
        result[i] = vocab.parse(item).v
    return result


def display_func(t, x):
    import base64
    import PIL.Image
    import cStringIO
    
    input_shape = (1, 14, 90)

    values = x.reshape(input_shape)
    values = values.transpose((1, 2, 0))
    values = (values + 1) / 2 * 255.
    values = values.astype('uint8')

    if values.shape[-1] == 1:
        values = values[:, :, 0]

    png = PIL.Image.fromarray(values)
    buffer = cStringIO.StringIO()
    png.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())

    display_func._nengo_html_ = '''
           <svg width="100%%" height="100%%" viewbox="0 0 %i %i">
           <image width="100%%" height="100%%"
                  xlink:href="data:image/png;base64,%s"
                  style="image-rendering: auto;">
           </svg>''' % (input_shape[2]*2, input_shape[1]*2, ''.join(img_str))


import ctn_benchmark
class Vision(ctn_benchmark.Benchmark):
    def params(self):
        self.default('number of V1 neurons', n_V1=80)
        self.default('number of inputs', n_input=2)
        self.default('dimensions', dimensions=32)
        self.default('fixed input', fixed_input=True)
        self.default('display from neurons', neuron_display=False)
    def model(self, p):
        names, images = load_images('images', count=p.n_input)
        
        model = nengo.Network()
        with model:
            rng = np.random.RandomState(p.seed)
            encoders = Gabor().generate(p.n_V1, (11, 11), rng=rng)
            encoders = Mask((14, 90)).populate(encoders, rng=rng, flatten=True)
            
            V1 = nengo.Ensemble(p.n_V1, images.shape[1], eval_points=images,
                                intercepts=nengo.dists.Choice([-0.5]),
                                max_rates=nengo.dists.Choice([100]),
                                encoders=encoders,
                                label = 'V1')
            
            vocab = spa.Vocabulary(p.dimensions)                    
            AIT = spa.State(p.dimensions, label='AIT', vocab=vocab)
            
            outputs = generate_vectors(names, vocab)
            f = nengo.utils.connection.target_function(images, outputs)
            visconn = nengo.Connection(V1, AIT.input, synapse=0.005,
                                        **f)
            
            if p.fixed_input:
                stim = nengo.Node(images[0], label='stim')
            else:
                def stim_func(t):
                    return images[int(t/0.1) % len(images)]
                stim = nengo.Node(stim_func, label='stim')
            nengo.Connection(stim, V1)
        
            
            display_node = nengo.Node(display_func, size_in=V1.size_out, 
                                        label='display_node')
            if p.neuron_display:
                nengo.Connection(V1, display_node)
            else:
                nengo.Connection(stim, display_node)
            
            
                               
        return model

if __name__ == "__builtin__":
    model = Vision().make_model(fixed_input=True, 
                                neuron_display=False,
                                dimensions=64,
                                n_V1=80,
                                n_input=5)