import nengo
import nengo.spa as spa
from nengo_extras.vision import Gabor, Mask

import numpy as np
import inspect, os, sys, time, csv, random
import matplotlib.pyplot as plt
import png
import itertools
import base64
import PIL.Image
import cStringIO
import ctn_benchmark


#display stimuli in gui, works for 28x90 (two words) and 14x90 (one word)
#t = time, x = vector of pixel values
def display_func(t, x):
    if np.size(x) > 14*90:
        input_shape = (1, 28, 90)
    else:
        input_shape = (1,14,90)

    values = x.reshape(input_shape) #back to 2d
    values = values.transpose((1, 2, 0))
    values = (values + 1) / 2 * 255. #0-255
    values = values.astype('uint8') #as ints

    if values.shape[-1] == 1:
        values = values[:, :, 0]

    #make png
    png_rep = PIL.Image.fromarray(values)
    buffer = cStringIO.StringIO()
    png_rep.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue())

    #html for nengo
    display_func._nengo_html_ = '''
           <svg width="100%%" height="100%%" viewbox="0 0 %i %i">
           <image width="100%%" height="100%%"
                  xlink:href="data:image/png;base64,%s"
                  style="image-rendering: auto;">
           </svg>''' % (input_shape[2]*2, input_shape[1]*2, ''.join(img_str))


class AssociationRecognition(ctn_benchmark.Benchmark):
    def params(self):
        self.default('dimensions for concepts', D=96)
        self.default('dimensions for mid-level', Dmid=48)
        self.default('dimensions for low-level motor', Dlow=32)
        self.default('subject number', subject=0)
        self.default('hand to use', hand='RIGHT')
        self.default('remove motor component', remove_motor=False)
        self.default('remove bg component', remove_bg=False)
        self.default('remove thal component', remove_thal=False)
        self.default('remove cortical component', remove_cortical=False)

    #load stimuli, subj=0 means a subset of the stims of subject 1 (no long words), works well with lower dims
    def load_stims(self, subj=0):

        #load files (targets/re-paired foils; short new foils; long new foils)
        #ugly, but this way we can use the original stimulus files
        stims = np.genfromtxt(self.cur_path + '/stims/S' + str(subj) + 'Other.txt', skip_header=True,
                              dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                     ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])
        stimsNFshort = np.genfromtxt(self.cur_path + '/stims/S' + str(subj) + 'NFShort.txt', skip_header=True,
                                     dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                            ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])


        if not(subj == 0):
            stimsNFlong = np.genfromtxt(self.cur_path + '/stims/S' + str(subj) + 'NFLong.txt', skip_header=True,
                                    dtype=[('probe', np.str_, 8), ('fan', 'int'), ('wordlen', np.str_, 8),
                                           ('item1', np.str_, 8), ('item2', np.str_, 8)], usecols=[1,2,3,4,5])

        #combine
        if not(subj == 0):
            stims = np.hstack((stims, stimsNFshort, stimsNFlong))
        else:
            stims = np.hstack((stims, stimsNFshort))

        stims = stims.tolist()

        #parse out different categories
        target_pairs = []
        target_words = []
        target_rpfoils = []
        new_foils = []
        items = []

        for i in stims:

            #fill items list with all words
            items.append(i[3])
            items.append(i[4])

            #get target pairs
            if i[0] == 'Target':
                target_pairs.append((i[3],i[4]))
                target_words.append(i[3])
                target_words.append(i[4])

            #make separate lists for targets/rp foils and new foils (for presenting experiment)
            if i[0] != 'NewFoil':
                target_rpfoils.append(i)
            else:
                new_foils.append(i)

        #remove duplicates
        items = np.unique(items).tolist()
        target_words = np.unique(target_words).tolist()

        self.target_pairs = target_pairs
        self.target_words = target_words
        self.items = items
        self.stims = stims
        self.target_rpfoils = target_rpfoils
        self.new_foils = new_foils

    #load images for vision
    def load_images(self):
        indir = self.cur_path + '/images/'
        files = os.listdir(indir)
        files2 = []

        #select only images for current item set
        for fn in files:
            if fn[-4:] == '.png' and (fn[:-4] in self.items):
                 files2.append(fn)

        X_train = np.empty(shape=(np.size(files2), 90*14),dtype='float32') #images x pixels matrix
        y_train_words = [] #words represented in X_train
        for i,fn in enumerate(files2):
                y_train_words.append(fn[:-4]) #add word

                #read in image and convert to 0-1 vector
                r = png.Reader(indir + fn)
                r = r.asDirect()
                image_2d = np.vstack(itertools.imap(np.uint8, r[2]))
                image_2d /= 255
                image_1d = image_2d.reshape(1,90*14)
                X_train[i] = image_1d

        #numeric labels for words (could present multiple different examples of words, would get same label)
        y_train = np.asarray(range(0,len(np.unique(y_train_words))))
        X_train = 2 * X_train - 1  # normalize to -1 to 1

        self.X_train = X_train
        self.y_train = y_train
        self.y_train_words = y_train_words


    #returns pixels of image representing item (ie METAL)
    def get_image(self, item):
        return self.X_train[self.y_train_words.index(item)]

    #returns pixels of image representing item (ie METAL)
    def get_vector(self, item):
        return self.train_targets[self.y_train_words.index(item)]

    #initialize vocabs
    def initialize_vocabs(self, p):
        #low level visual representations
        self.vocab_vision = nengo.spa.Vocabulary(p.Dmid,max_similarity=.5)
        for name in self.y_train_words:
            self.vocab_vision.parse(name)
        self.train_targets = self.vocab_vision.vectors


        #word concepts - has all concepts, including new foils
        self.vocab_concepts = spa.Vocabulary(p.D, max_similarity=0.2)
        for i in self.y_train_words:
            self.vocab_concepts.parse(i)
        self.vocab_concepts.parse('ITEM1')
        self.vocab_concepts.parse('ITEM2')


        #vision-concept mapping between vectors
        self.vision_mapping = np.zeros((p.D, p.Dmid))
        for word in self.y_train_words:
            self.vision_mapping += np.outer(self.vocab_vision.parse(word).v, self.vocab_concepts.parse(word).v).T

        #vocab with learned words
        self.vocab_learned_words = self.vocab_concepts.create_subset(self.target_words)

        #vocab with learned pairs
        self.list_of_pairs = []
        for item1, item2 in self.target_pairs:
            #vocab_learned_pairs.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2)) #think this can go, but let's see
            #vocab_learned_pairs.add('%s_%s' % (item1,item2), vocab_learned_pairs.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2)))
            self.vocab_concepts.add('%s_%s' % (item1,item2), self.vocab_concepts.parse('%s*ITEM1 + %s*ITEM2' % (item1, item2))) #add pairs to concepts to use same vectors
            self.list_of_pairs.append('%s_%s' % (item1,item2)) #keep list of pairs notation
        self.vocab_learned_pairs = self.vocab_concepts.create_subset(self.list_of_pairs) #get only pairs
        #print vocab_learned_pairs.keys

        #motor vocabs, just for sim calcs
        self.vocab_motor = spa.Vocabulary(p.Dmid) #different dimension to be sure, upper motor hierarchy
        self.vocab_motor.parse('LEFT+RIGHT+INDEX+MIDDLE')

        self.vocab_fingers = spa.Vocabulary(p.Dlow) #direct finger activation
        self.vocab_fingers.parse('L1+L2+R1+R2')

        #map higher and lower motor
        self.motor_mapping = np.zeros((p.Dlow, p.Dmid))
        self.motor_mapping += np.outer(self.vocab_motor.parse('LEFT+INDEX').v, self.vocab_fingers.parse('L1').v).T
        self.motor_mapping += np.outer(self.vocab_motor.parse('LEFT+MIDDLE').v, self.vocab_fingers.parse('L2').v).T
        self.motor_mapping += np.outer(self.vocab_motor.parse('RIGHT+INDEX').v, self.vocab_fingers.parse('R1').v).T
        self.motor_mapping += np.outer(self.vocab_motor.parse('RIGHT+MIDDLE').v, self.vocab_fingers.parse('R2').v).T

        #goal vocab
        self.vocab_goal = spa.Vocabulary(p.Dlow)
        self.vocab_goal.parse('DO_TASK')
        self.vocab_goal.parse('RECOG')
        self.vocab_goal.parse('RESPOND')
        self.vocab_goal.parse('END')

        #attend vocab
        self.vocab_attend = self.vocab_concepts.create_subset(['ITEM1', 'ITEM2'])


    def model(self, p):
        self.cur_path = '.'
        self.load_stims(p.subject)
        self.load_images()
        self.initialize_vocabs(p)

        trial_info=('Target', 1, 'Short', 'METAL', 'SPARK')

        #word presented in current trial
        item1 = trial_info[3]
        item2 = trial_info[4]

        #returns images of current words
        def present_pair(t):
            im1 = self.get_image(item1)
            im2 = self.get_image(item2)
            return np.hstack((im1, im2))

        #returns image 1 <100 ms, otherwise image 2
        def present_item(t):
            if t < .1:
                return self.get_vector(item1)
            else:
                return self.get_vector(item2)

        def present_item2(t,output_attend):

            similarities = [np.dot(output_attend, self.vocab_attend['ITEM1'].v),
                            np.dot(output_attend, self.vocab_attend['ITEM2'].v)]
            #print similarities

            ret_ima = np.zeros(1260)
            if similarities[0] > .5:
                ret_ima = self.get_image(item1)
            elif similarities[1] > .5:
                ret_ima = self.get_image(item2)

            return ret_ima


        model = spa.SPA(seed=p.seed)

        if p.backend == 'nengo_spinnaker':
            import nengo_spinnaker
            nengo_spinnaker.add_spinnaker_params(model.config)


        with model:

            #display current stimulus pair (not part of model)
            #model.pair_input = nengo.Node(present_pair)
            #model.pair_display = nengo.Node(display_func, size_in=model.pair_input.size_out)  # to show input
            #nengo.Connection(model.pair_input, model.pair_display, synapse=None)


            # control
            model.control_net = nengo.Network()
            with model.control_net:
                model.attend = spa.State(p.D, vocab=self.vocab_attend, feedback=.5)  # vocab_attend
                model.goal = spa.State(p.D, self.vocab_goal, feedback=1)  # current goal Dlow
                model.target_hand = spa.State(p.Dmid, vocab=self.vocab_motor, feedback=1)


            ### vision ###

            # set up network parameters
            n_vis = self.X_train.shape[1]  # nr of pixels, dimensions of network
            n_hid = 1000  # nr of gabor encoders/neurons

            # random state to start
            rng = np.random.RandomState(seed=p.seed)
            encoders = Gabor().generate(n_hid, (11, 11), rng=rng)  # gabor encoders, 11x11 apparently, why?
            encoders = Mask((14, 90)).populate(encoders, rng=rng,
                                               flatten=True)  # use them on part of the image

            model.visual_net = nengo.Network()
            with model.visual_net:

                #represent currently attended item
                model.attended_item = nengo.Node(present_item, label='attended_item')
                if p.backend == 'nengo_spinnaker':
                    model.config[model.attended_item].function_of_time = True
                #model.attended_item = nengo.Node(present_item2,size_in=model.attend.output.size_out)
                #nengo.Connection(model.attend.output, model.attended_item)

                #model.vision_gabor = nengo.Ensemble(n_hid, n_vis, eval_points=X_train,
                #                                        neuron_type=nengo.LIFRate(),
                #                                        intercepts=nengo.dists.Choice([-0.5]),
                #                                        max_rates=nengo.dists.Choice([100]),
                #                                        encoders=encoders)

                model.visual_representation = spa.State(p.Dmid)
                #model.visual_representation = nengo.Ensemble(n_hid, dimensions=Dmid)
                nengo.Connection(model.attended_item, model.visual_representation.input,
                                        synapse=0.005)

                #model.visconn = nengo.Connection(model.vision_gabor, model.visual_representation, synapse=0.005,
                #                                eval_points=X_train, function=train_targets,
                #                                solver=nengo.solvers.LstsqL2(reg=0.01))
                #nengo.Connection(model.attended_item, model.vision_gabor, synapse=None)

                # display attended item
                #model.display_attended = nengo.Node(display_func, size_in=model.attended_item.size_out)  # to show input
                #nengo.Connection(model.attended_item, model.display_attended, synapse=None)





            # concepts
            model.concepts = spa.AssociativeMemory(self.vocab_concepts,wta_output=True,wta_inhibit_scale=1)
            nengo.Connection(model.visual_representation.output, model.concepts.input, transform=self.vision_mapping)

            # pair representation
            model.vis_pair = spa.State(p.D, vocab=self.vocab_concepts, feedback=2)

            model.dm_learned_words = spa.AssociativeMemory(self.vocab_learned_words) #familiarity should be continuous over all items, so no wta
            nengo.Connection(model.dm_learned_words.output,model.dm_learned_words.input,transform=.5,synapse=.01)

            model.familiarity = spa.State(1,feedback_synapse=.01) #no fb syn specified
            nengo.Connection(model.dm_learned_words.am.elem_output,model.familiarity.input, #am.element_output == all outputs, we sum
                             transform=.8*np.ones((1,model.dm_learned_words.am.elem_output.size_out)))

            model.dm_pairs = spa.AssociativeMemory(self.vocab_learned_pairs, input_keys=self.list_of_pairs,wta_output=True)
            nengo.Connection(model.dm_pairs.output,model.dm_pairs.input,transform=.5)

            #this works:
            model.representation = spa.AssociativeMemory(self.vocab_learned_pairs, input_keys=self.list_of_pairs, wta_output=True)
            nengo.Connection(model.representation.output, model.representation.input, transform=2)
            model.rep_filled = spa.State(1,feedback_synapse=.005) #no fb syn specified
            nengo.Connection(model.representation.am.elem_output,model.rep_filled.input, #am.element_output == all outputs, we sum
                             transform=.8*np.ones((1,model.representation.am.elem_output.size_out)),synapse=0.005)

            #this doesn't:
            #model.representation = spa.State(D,feedback=1)
            #model.rep_filled = spa.State(1,feedback_synapse=.005) #no fb syn specified
            #nengo.Connection(model.representation.output,model.rep_filled.input, #am.element_output == all outputs, we sum
            #                 transform=.8*np.ones((1,model.representation.output.size_out)),synapse=0)


            # this shouldn't really be fixed I think
            model.comparison = spa.Compare(p.D, vocab=self.vocab_concepts)


            if not p.remove_motor:
                #motor
                model.motor_net = nengo.Network()
                with model.motor_net:

                    #input multiplier
                    model.motor_input = spa.State(p.Dmid,vocab=self.vocab_motor)

                    #higher motor area (SMA?)
                    model.motor = spa.State(p.Dmid, vocab=self.vocab_motor,feedback=1)

                    #connect input multiplier with higher motor area
                    nengo.Connection(model.motor_input.output,model.motor.input,synapse=.1,transform=10)

                    #finger area
                    model.fingers = spa.AssociativeMemory(self.vocab_fingers, input_keys=['L1', 'L2', 'R1', 'R2'], wta_output=True)

                    #conncetion between higher order area (hand, finger), to lower area
                    nengo.Connection(model.motor.output, model.fingers.input, transform=.4*self.motor_mapping)

                    #finger position (spinal?)
                    model.finger_pos = nengo.networks.EnsembleArray(n_neurons=50, n_ensembles=4)
                    nengo.Connection(model.finger_pos.output, model.finger_pos.input, synapse=0.1, transform=0.3) #feedback

                    #connection between finger area and finger position
                    nengo.Connection(model.fingers.am.elem_output, model.finger_pos.input, transform=np.diag([0.55, .53, .57, .55])) #fix these



            if not p.remove_bg:

                model.bg = spa.BasalGanglia(
                    spa.Actions(
                        'dot(goal,DO_TASK)-.5 --> dm_learned_words=vis_pair, goal=RECOG, attend=ITEM1',
                        'dot(goal,RECOG)+dot(attend,ITEM1)+familiarity-2 --> goal=RECOG2, dm_learned_words=vis_pair, attend=ITEM2',#'vis_pair=ITEM1*concepts',
                        'dot(goal,RECOG)+dot(attend,ITEM1)+(1-familiarity)-2 --> goal=RECOG2, attend=ITEM2', #motor_input=1.5*target_hand+MIDDLE,
                        'dot(goal,RECOG2)+dot(attend,ITEM2)+familiarity-1.3 --> goal=RECOLLECTION,dm_pairs = 2*vis_pair, representation=3*dm_pairs',# vis_pair=ITEM2*concepts',
                        'dot(goal,RECOG2)+dot(attend,ITEM2)+(1-familiarity)-1.3 --> goal=RESPOND, motor_input=1.0*target_hand+MIDDLE',
                        'dot(goal,RECOLLECTION) - .5 --> goal=RECOLLECTION, representation=2*dm_pairs',
                        'dot(goal,RECOLLECTION) + 2*rep_filled - 1.3 --> goal=COMPARE_ITEM1, attend=ITEM1, comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                        'dot(goal,COMPARE_ITEM1) + rep_filled + comparison -1 --> goal=COMPARE_ITEM2, attend=ITEM2, comparison_A = 2*vis_pair',#comparison_B = 2*representation*~attend',
                        'dot(goal,COMPARE_ITEM1) + rep_filled + (1-comparison) -1 --> goal=RESPOND,motor_input=1.0*target_hand+MIDDLE',#comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                        'dot(goal,COMPARE_ITEM2) + rep_filled + comparison - 1 --> goal=RESPOND,motor_input=1.0*target_hand+INDEX',#comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                        'dot(goal,COMPARE_ITEM2) + rep_filled + (1-comparison) -1 --> goal=RESPOND,motor_input=1.0*target_hand+MIDDLE',#comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',

                        'dot(goal,RESPOND) + comparison - 1 --> goal=RESPOND, motor_input=1.0*target_hand+INDEX', #comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',
                        'dot(goal,RESPOND) + (1-comparison) - 1 --> goal=RESPOND, motor_input=1.0*target_hand+MIDDLE', #comparison_A = 2*vis_pair,comparison_B = 2*representation*~attend',

                        # 'dot(goal,RECOLLECTION) + (1 - dot(representation,vis_pair)) - 1.3 --> goal=RESPOND, motor_input=1.0*target_hand+MIDDLE',
                        'dot(goal,RESPOND)+dot(motor,MIDDLE+INDEX)-1.0 --> goal=END',
                        'dot(goal,END) --> goal=END',
                        #'.6 -->',

                        #possible to match complete buffer, ie is representation filled?

                    ))
                if not p.remove_thal:
                    model.thalamus = spa.Thalamus(model.bg)

            if not p.remove_cortical:
                model.cortical = spa.Cortical( # cortical connection: shorthand for doing everything with states and connections
                    spa.Actions(
                      #  'motor_input = .04*target_hand',
                        #'dm_learned_words = .8*concepts', #.5
                        #'dm_pairs = 2*stimulus'
                        'vis_pair = 2*attend*concepts+concepts',
                        'comparison_A = 2*vis_pair',
                        'comparison_B = 2*representation*~attend',

                    ))


            if not p.remove_motor:
                self.pr_motor_pos = nengo.Probe(model.finger_pos.output,synapse=.01) #raw vector (dimensions x time)
                self.pr_motor = nengo.Probe(model.fingers.output,synapse=.01)
                self.pr_motor1 = nengo.Probe(model.motor.output, synapse=.01)

            #input
            model.input = spa.Input(goal=lambda t: 'DO_TASK' if t < 0.05 else '0',
                                    target_hand=p.hand,
                                    #attend=lambda t: 'ITEM1' if t < 0.1 else 'ITEM2',
                                    )
            if p.backend == 'nengo_spinnaker':
                for n in model.input.nodes:
                    model.config[n].function_of_time = True
        return model

    def evaluate(self, p, sim, plt):
        sim.run(0.5)

        if plt:
            plt.plot(sim.trange(), sim.data[self.pr_motor_pos])

        return {}


if __name__ == '__main__':
    AssociationRecognition().run()
