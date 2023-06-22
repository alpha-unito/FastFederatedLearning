//
// Created by gmittone on 6/22/23.
//

#ifndef FASTFEDERATEDLEARNING_SERIALIZE_HPP
#define FASTFEDERATEDLEARNING_SERIALIZE_HPP

#include <torch/serialize.h>

#include "net.hpp"

void save_to_stream(std::ostringstream &ostream, std::vector <at::Tensor> const &data) {
    std::ostringstream stream;
    torch::save(data, stream);
    std::string buffer = stream.str();
    ostream << buffer.length() << buffer;
}

void read_from_stream(std::istringstream &istream, std::vector <at::Tensor> &data) {
    long size;
    istream >> size;

    char *buffer = new char[size * sizeof(char)];
    istream.read(buffer, size);

    std::string string_buffer(buffer, size);
    // Creare uno string buffer è necessario perché char* torch load lo interpreta come filename
    // TODO: fare overloading di torch::load?
    std::istringstream stream(string_buffer);
    torch::load(data, stream);

    delete buffer;
}

// cereal based serialization does not work
template<class Archive>
void save(Archive &archive, StateDict const &m) {
    std::ostringstream oss;

    save_to_stream(oss, m.parameters_data);
    save_to_stream(oss, m.buffers_data);

    archive(oss.str());
}

template<class Archive>
void load(Archive &archive, StateDict &m) {
    std::string oss;
    archive(oss);
    std::istringstream iss(oss);

    read_from_stream(iss, m.parameters_data);
    read_from_stream(iss, m.buffers_data);
}

#endif //FASTFEDERATEDLEARNING_SERIALIZE_HPP
