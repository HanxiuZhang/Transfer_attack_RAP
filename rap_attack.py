import torch

def rap_attack():
    for k in range(0, num_batches):
        batch_size_cur = min(arg.batch_size, len(image_id_list) - k * arg.batch_size)
        X_ori = torch.zeros(batch_size_cur, 3, img_size, img_size).to(device)
        delta = torch.zeros_like(X_ori, requires_grad=True).to(device)
        for i in range(batch_size_cur):
            X_ori[i] = trn(Image.open(input_path + image_id_list[k * arg.batch_size + i] + '.png'))
        labels = torch.tensor(label_ori_list[k * arg.batch_size:k * arg.batch_size + batch_size_cur]).to(device)
        target_labels = torch.tensor(label_tar_list[k * arg.batch_size:k * arg.batch_size + batch_size_cur]).to(device)
        grad_pre = 0
        prev = float('inf')

        if arg.random_start:
            # Starting at a uniformly random point
            delta.requires_grad_(False)
            delta = delta + torch.empty_like(X_ori).uniform_(-epsilon/255, epsilon/255)
            delta = torch.clamp(X_ori+delta, min=0, max=1) - X_ori
            delta.requires_grad_(True)

        logging(50*"#")
        logging("starting :{} batch".format(k+1))
        

        for t in range(arg.max_iterations):
            if t < arg.transpoint:
                adv_activate = 0
            else:
                if arg.adv_perturbation:
                    adv_activate = 1
                else:
                    adv_activate = 0
            grad_list = []

            for q in range(arg.m1):
                delta.requires_grad_(False)

                if arg.strength == 0:
                    X_addin = torch.zeros_like(X_ori).to(device)
                else:
                    X_addin = torch.zeros_like(X_ori).to(device)
                    random_labels = torch.zeros(batch_size_cur).to(device)
                    stop = False
                    while stop == False:
                        random_indices = np.random.randint(0, 1000, batch_size_cur)
                        for i in range(batch_size_cur):
                            X_addin[i] = trn(Image.open(input_path + image_id_list[random_indices[i]] + '.png'))
                            random_labels[i] = label_ori_list[random_indices[i]]
                        if torch.sum(random_labels==labels).item() == 0:
                            stop = True
                    X_addin = arg.strength * X_addin
                    X_addin = torch.clamp(X_ori+delta+X_addin, min=0, max=1) - (X_ori+delta)
                
                if arg.SI:

                    if adv_activate:
                        top_values_1, top_indices_1 = model_source(norm(X_ori+delta+X_addin)).topk(arg.m1+1, dim=1, largest=True, sorted=True)
                        
                        if arg.adv_targeted:
                            label_pred = labels
                        else:
                            label_pred = target_labels

                        X_advaug = pgd(model_source, X_ori+delta+X_addin, label_pred, arg.adv_targeted, arg.adv_epsilon, arg.adv_steps, arg.adv_alpha)
                        X_aug = X_advaug - (X_ori+delta+X_addin)

                    else:
                        X_aug = torch.zeros_like(X_ori).to(device)

                delta.requires_grad_(True)

                for j in range(arg.m2):

                    if not arg.SI:
                        delta.requires_grad_(False)

                        if adv_activate:
                            top_values_2, top_indices_2 = model_source(norm(X_ori+delta+X_addin)).topk(arg.m2+1, dim=1, largest=True, sorted=True)
                            
                            if arg.adv_targeted:
                                label_pred = labels
                            else:
                                label_pred = target_labels
                            
                            X_advaug = pgd(model_source, X_ori+delta+X_addin, label_pred, arg.adv_targeted, arg.adv_epsilon, arg.adv_steps, arg.adv_alpha)
                            X_aug = X_advaug - (X_ori+delta+X_addin)

                        else:
                            X_aug = torch.zeros_like(X_ori).to(device)
                        delta.requires_grad_(True)

                    if arg.DI:  # DI
                        if arg.SI:
                            logits = model_source(norm(DI((X_ori + delta + X_addin + X_aug )/2**j)))
                        else:
                            logits = model_source(norm(DI(X_ori + delta + X_addin + X_aug )))
                    else:
                        if arg.SI:
                            logits = model_source(norm((X_ori + delta + X_addin + X_aug )/2**j))
                        else:
                            logits = model_source(norm(X_ori + delta + X_addin + X_aug ))

                    if arg.loss_function == 'CE':
                        loss_func = nn.CrossEntropyLoss(reduction='sum')
                        if arg.targeted:
                            loss = loss_func(logits, target_labels)
                        else:
                            loss = -1 * loss_func(logits, labels)

                    elif arg.loss_function == 'MaxLogit':
                        if arg.targeted:
                            real = logits.gather(1,target_labels.unsqueeze(1)).squeeze(1)
                            loss = -1 * real.sum()
                        else:
                            real = logits.gather(1,labels.unsqueeze(1)).squeeze(1)
                            loss = real.sum()

                    loss.backward()
                    grad_cc = delta.grad.clone().to(device)

                    if arg.TI:  # TI
                        grad_cc = F.conv2d(grad_cc, gaussian_kernel, bias=None, stride=1, padding=(2, 2), groups=3)
                    grad_list.append(grad_cc)
                    delta.grad.zero_()

            grad_c = 0

            for j in range(arg.m1 * arg.m2):
                grad_c += grad_list[j]
            grad_c = grad_c / (arg.m1 * arg.m2)

            if arg.MI:  # MI
                grad_c = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre

            grad_pre = grad_c
            delta.data = delta.data - lr * torch.sign(grad_c)
            delta.data = delta.data.clamp(-epsilon / 255, epsilon / 255)
            delta.data = ((X_ori + delta.data).clamp(0, 1)) - X_ori

            if t % 10 == 9:
                if arg.targeted:
                    pos[0, t // 10] = pos[0, t // 10] + sum(torch.argmax(model_1(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
                    pos[1, t // 10] = pos[1, t // 10] + sum(torch.argmax(model_2(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
                    pos[2, t // 10] = pos[2, t // 10] + sum(torch.argmax(model_3(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
                    pos[3, t // 10] = pos[3, t // 10] + sum(torch.argmax(model_4(norm(X_ori + delta)), dim=1) == target_labels).cpu().numpy()
                else:
                    pos[0, t // 10] = pos[0, t // 10] + sum(torch.argmax(model_1(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()
                    pos[1, t // 10] = pos[1, t // 10] + sum(torch.argmax(model_2(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()
                    pos[2, t // 10] = pos[2, t // 10] + sum(torch.argmax(model_3(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()
                    pos[3, t // 10] = pos[3, t // 10] + sum(torch.argmax(model_4(norm(X_ori + delta)), dim=1) != labels).cpu().numpy()

                logging(str(pos))
                logging(30*"#")



            if t == (1-1):
                X_adv_10[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
            if t == (50-1):
                X_adv_50[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
            if t == (100-1):
                X_adv_100[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
            if t == (200-1):
                X_adv_200[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
            if t == (300-1):
                X_adv_300[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
            if t == (400-1):
                X_adv_400[fixing_point:fixing_point+batch_size_cur] = (X_ori + delta).clone().detach().cpu()
        
        fixing_point += batch_size_cur
        logging(50*"#")