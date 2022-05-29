import time
import torch

from helpers import list_of_distances, make_one_hot
from settings import img_size
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
def _train_or_test(model, dataloader, optimizer=None, class_specific=False, use_l1_mask=True,
                   coefs=None, log=print, is_attn=True, k=10, ppnet=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    for i, (image, label) in enumerate(dataloader):
        attn = image[:, 3, :, :].cuda()
        input = image[:, :3, :, :].cuda()
        target = label.cuda()

        batch_size = attn.size()[0]

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward

            output, min_distances, activations, patterns = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost for class-specific visual words
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            if class_specific:
                total_separation_cost += separation_cost.item()
                total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step

        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1
                          )
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1
                          )
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1


            if is_attn:
                # calculate the attention coverage loss
                largest = torch.topk(min_distances, k, dim=1, largest=False).indices
                largest = largest.unsqueeze(-1).unsqueeze(-1).expand(largest.shape[0], k, patterns.shape[2], patterns.shape[3])
                patterns = torch.gather(patterns, dim=1, index = largest)
                patterns = F.interpolate(patterns, [img_size, img_size], mode='bilinear', align_corners=True)
                patterns = torch.max(patterns, dim=1)[0]
                patterns = patterns.view(batch_size, -1)
                patterns = patterns / (patterns.max(dim=1, keepdim=True).values + 1e-5)
                attn_loss_f = torch.nn.MSELoss(reduction='mean')
                attn_loss = attn_loss_f(patterns, attn.view(batch_size, -1))
                loss = loss + 10 * attn_loss 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(model.module.last_layer.weight.norm(p=1).item()))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    return n_correct / n_examples


def train(model, dataloader, optimizer, class_specific=True, coefs=None, log=print, train_attn=False, ppnet=None):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()

    if train_attn:
        print("True")
        return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                            class_specific=class_specific, coefs=coefs, log=log, is_attn=True, ppnet=ppnet)
        
    else:
        # print("False")
        # return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
        #                     class_specific=class_specific, coefs=coefs, log=log, is_attn=False)
        print("False")
        return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                            class_specific=class_specific, coefs=coefs, log=log, is_attn=True)    

def train_warm(model, dataloader, optimizer, class_specific=True, coefs=None, log=print):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    print("False")
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          class_specific=class_specific, coefs=coefs, log=log, is_attn=False)

def test(model, dataloader, class_specific=True, log=print):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          class_specific=class_specific, log=log)


def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')


def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')


def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
