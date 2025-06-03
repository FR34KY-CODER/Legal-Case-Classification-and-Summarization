
import pandas as pd
ik = pd.read_csv(r'C:\Users\FR34K\Desktop\Coding\BTP\Legal_Case_Dataset_Final.csv')

# Specify the columns to combine in the desired order
columns_to_combine = ["Fact", "Issue", "Petitioner's Argument", "Respondent's Argument",
    "Precedent Analysis", "Analysis of the law", "Court's Reasoning", "Conclusion"]

# Create the new column by concatenating the text data row-wise
ik['Case Content'] = ik[columns_to_combine].fillna('').astype(str).agg(''.join, axis=1)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# ==== 1) MODEL COMPONENTS ==== #

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_lengths):
        # src: [B, src_len]
        embedded = self.dropout(self.embedding(src))  # [B, src_len, emb_dim]
        # pack for variable-length
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # [B, src_len, hid_dim*2]
        return outputs, hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        # We’ll combine decoder hidden + encoder outputs
        self.attn = nn.Linear(hid_dim + hid_dim*2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
    
    def forward(self, dec_hidden, enc_outputs, mask):
        # dec_hidden: [B, hid_dim]
        # enc_outputs: [B, src_len, hid_dim*2]
        B, src_len, _ = enc_outputs.size()
        dec_hidden = dec_hidden.unsqueeze(1).repeat(1, src_len, 1)  # [B, src_len, hid_dim]
        energy = torch.tanh(self.attn(torch.cat((dec_hidden, enc_outputs), dim=2)))  # [B, src_len, hid_dim]
        scores = self.v(energy).squeeze(2)  # [B, src_len]
        scores = scores.masked_fill(mask == 0, -1e9)  # mask paddings
        return torch.softmax(scores, dim=1)  # [B, src_len]

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, attention):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(
            emb_dim + hid_dim*2,
            hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            batch_first=True
        )
        self.attention = attention
        self.fc_out = nn.Linear(hid_dim*3 + emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_tok, dec_hidden, dec_cell, enc_outputs, mask):
        # input_tok: [B]  (next input token id)
        B = input_tok.size(0)
        embedded = self.dropout(self.embedding(input_tok)).unsqueeze(1)  # [B,1,emb_dim]
        
        # compute attention weights
        a = self.attention(dec_hidden[-1], enc_outputs, mask)  # [B, src_len]
        a = a.unsqueeze(1)  # [B,1,src_len]
        # weighted sum of encoder outputs
        weighted = torch.bmm(a, enc_outputs)  # [B,1,hid_dim*2]
        
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [B,1,emb_dim+hid*2]
        output, (dec_hidden, dec_cell) = self.rnn(rnn_input, (dec_hidden, dec_cell))
        
        output = output.squeeze(1)    # [B, hid_dim]
        weighted = weighted.squeeze(1)  # [B, hid_dim*2]
        embedded = embedded.squeeze(1)  # [B, emb_dim]
        
        preds = self.fc_out(torch.cat((output, weighted, embedded), dim=1))  # [B, vocab_size]
        return preds, dec_hidden, dec_cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, pad_idx):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_idx = pad_idx
    
    def create_mask(self, src):
        # mask=1 for non-pad
        return (src != self.pad_idx).to(self.device)
    
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        # src: [B, src_len], trg: [B, trg_len]
        B, trg_len = trg.size()
        vocab_size = self.decoder.fc_out.out_features
        
        # tensor to store outputs
        outputs = torch.zeros(B, trg_len, vocab_size, device=self.device)
        
        enc_outputs, hidden, cell = self.encoder(src, src_lengths)
        
        # concat bidirectional hidden states
        hidden = self._combine_directions(hidden)
        cell   = self._combine_directions(cell)
        
        # first input to decoder is <sos>
        input_tok = trg[:,0]
        mask = self.create_mask(src)
        
        for t in range(1, trg_len):
            preds, hidden, cell = self.decoder(input_tok, hidden, cell, enc_outputs, mask)
            outputs[:,t] = preds
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = preds.argmax(1)
            input_tok = trg[:,t] if teacher_force else top1
        
        return outputs
    
    def _combine_directions(self, h):
        # h: [n_layers*2, B, hid_dim]
        n_layers = h.size(0) // 2
        # reshape -> [n_layers, 2, B, hid_dim]
        h = h.view(n_layers, 2, h.size(1), h.size(2))
        # concat forward & backward -> [n_layers, B, hid_dim*2]
        return torch.cat((h[:,0], h[:,1]), dim=2)

# ==== 2) DATASET ==== #

class SummarizationDataset(Dataset):
    def __init__(self, df):
        # df must have 'input_ids' & 'target_ids' as lists/tensors
        self.src = [torch.tensor(ids, dtype=torch.long) for ids in df['input_ids']]
        self.trg = [torch.tensor(ids, dtype=torch.long) for ids in df['target_ids']]

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.trg[idx]

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    src_padded = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    trg_padded = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
    return src_padded, torch.tensor(src_lens), trg_padded

# ==== 3) TRAINING LOOP ==== #

if __name__ == "__main__":
    # Hyperparams & device
    DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VOCAB_SZ  = 30000
    EMB_DIM   = 256
    HID_DIM   = 512
    N_LAYERS  = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5
    BATCH_SZ  = 8
    N_EPOCHS  = 10
    LEARNING_RATE = 1e-3
    global PAD_IDX
    PAD_IDX = 0  # adjust to your tokenizer’s pad token ID

    # 1) DataLoader
    dataset = SummarizationDataset(ik)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SZ,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 2) Build model
    attn = Attention(HID_DIM)
    enc  = Encoder(VOCAB_SZ, EMB_DIM, HID_DIM//2, N_LAYERS, ENC_DROPOUT)  # hid_dim//2 per direction
    dec  = Decoder(VOCAB_SZ, EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, DEVICE, PAD_IDX).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # 3) Training
    for epoch in range(1, N_EPOCHS+1):
        model.train()
        epoch_loss = 0
        for src, src_lens, trg in loader:
            src, src_lens, trg = src.to(DEVICE), src_lens.to(DEVICE), trg.to(DEVICE)
            optimizer.zero_grad()
            output = model(src, src_lens, trg)  
            # output: [B, trg_len, vocab_sz]
            output_dim = output.shape[-1]
            # skip first token
            out = output[:,1:].reshape(-1, output_dim)
            tgt = trg[:,1:].reshape(-1)
            loss = criterion(out, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch} | Loss: {epoch_loss/len(loader):.4f}")
