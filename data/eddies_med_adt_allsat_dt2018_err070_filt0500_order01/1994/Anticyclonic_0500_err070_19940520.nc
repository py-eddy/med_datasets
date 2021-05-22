CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�G�z�H       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�s7   max       P�:b       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       >�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @E�=p��
     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vbz�G�     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q@           �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ɯ        max       @���           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �ě�   max       >Kƨ       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�K�   max       B,�R       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�:z   max       B,z�       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�Z�   max       C��       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��8   max       C��~       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�s7   max       P�`�       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�g��	k�   max       ?�ݗ�+j�       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �e`B   max       >�       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @E��z�H     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @vbz�G�     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @Q@           �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ɯ        max       @��@           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?R   max         ?R       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?}��Z�   max       ?�����$     �  Y|   $   	         	            @         R         +                     -                     C         6               *   [                     {      !   K            
         	         \               /   �   D   O�}�N ��N��'N."kN\M�s7M�<�N,?P��SN*i�N?j�P�:bNbRN�^�P�N�XkO A	O�dEN���No�PN�ˣP
��O���P*�nPO�PO,tM��P�qDO�0�NǦO��OY��N�pKP ��N��OzsP� O2A�Nk,YN\�O�:�N�VN�@hP\�6O�Om��P1�<Oc�N|Y�N�UN�*�N8�N��uN�^�N��N��PDK�NɣNB��M�z�N�Y�O��*O��O���Nk���ě��e`B�49X�#�
�o��o��o%   :�o;o;o;o;o;o;�o;�o;��
;��
;ě�<o<t�<#�
<D��<D��<u<�C�<�C�<�t�<�t�<��
<��
<��
<�1<�9X<�9X<ě�<ě�<�h<�h<��=o=o=o=o=o=+=C�=C�=\)=t�=�w=#�
='�=,1=0 �=0 �=49X=L��=L��=L��=]/=ix�=q��=�o=�1>����������������������������������[[\hjt�������th[[[[[��������������������������������������������������������������������������������IN[`gigb[ZVNIIIIIIII<Hanz��������aH<"���������������������������������������������BXtnk[95����=EHUaanonaUH========#/0<CIRRLI?<0-)#����
/7GJG</#
������zw|����������������caghnt���������tigcccb^h��������������jc��������������������}�������������}}}}}}��������������������"/;?GXZ]_ZTH/"�����)5=:5)�BJS[gtvlsmZN5��������������������	+5BNJKJEHMB5)ssx{}{������������ts��������������������������);DA6)��������������������������"#)6?BOYUOB6+)(���������������������������������������������������������������
/TardidU<#
�������!"/<HU^bcaZTUHB<8/%!67=>COht�������th[B6adht�������������tha��������������������mmqz~����zqmmmmmmmmm#$%)05BJOQWYQPNM;1*#NMNNS[^gntxytsg[NNNNknnz��������������nk��������/0&���������������   �������� !')5BJNT\bf[OB5) MSX`knmt���������g[M������!+0/*������������������������V_`abnnswniaVVVVVVVV/,-05BDLNT[_[NMB85//���
	��������������������������������������������������������������"/;;==;/"y}���������������~{y����

������������!#+/563/)#!!!!!!!!UUIGILU[bbbWUUUUUUUU�������������������� �	6ADIJHC@6) ��������
 
�������������������������������������������������ûɻһѻû��������l�_�L�S�_�v�������������������������������������������@�B�L�V�Y�_�b�Y�Y�U�L�@�=�:�8�@�@�@�@�@�'�)�3�7�6�4�3�0�'�!����$�'�'�'�'�'�'������������������������ùùϹ۹ܹܹܹϹϹù����ùùùùùùù�²¿��������¿²±±²²²²²²²²²²�������������������������������������������������������r�Z�N�����ݿڿҿֿ���a�l�m�z���z�z�y�m�a�[�\�a�a�a�a�a�a�a�a�tāćċąĄā�~�w�t�t�o�t�t�t�t�t�t�t�t���#�:�D�{ŋŦŔŔŇ�<�����Ĳġĳ�������a�m�o�y�m�m�e�a�_�[�X�X�a�a�a�a�a�a�a�a���������������������r�l�f�e�c�f�r�z��m�������������y�`�G�;�.�'�&�+�;�G�T�`�m�������ĽɽνнннĽ������������������������������������������������������޾Z�f�s�|�����w�f�Z�U�M�I�A�4�1�'�.�4�M�Z�ѿݿ����������ݿԿѿɿɿѿѿѿѿѿ��;�@�H�N�I�I�H�;�9�5�/�,�/�4�;�;�;�;�;�;��*�6�;�C�O�O�\�O�C�6�*� ��������/�T�a�h�r�u�s�m�T�;��	���������������/�h�uƁƎƗƚƞƚƙƕƈ��u�h�\�J�@�:�A�h�������D�I�V�m�w�o�V�0���������ƾ���̿"�.�T�`�m�|�|�j�T�;���	��������"�Z�f�k�s�{����s�f�Z�M�A�4�.�*�3�8�B�M�Z��*�6�C�O�\�d�X�O�M�C�6�*��������m�z��������z�m�g�i�m�m�m�m�m�m�m�m�m�m�����"�/�A�X�X�L�;�"�	���������������������������������������s�[�R�Z�s���������˿ݿ��������������ݿܿ׿տѿпѿڿݿ�ù��������������ùìàÓÇÄÁÇÊÓìùŠŭŹſ������������ŹŭŠŗŔŊŋŔŗŠ�6�B�G�F�F�F�B�B�6�)�$�%�)�.�6�6�6�6�6�6�"�,�1�*�-�"��	����������������	���"����������ùøù�����������������������������������������ùþ������������ܻ���'�@�[�k�e�F�B�'��&�%���޻�ֻ���#�(�.�9�<�B�B�<�;�/�#���
���	�����������������������������������������޿ѿѿݿ�ݿֿѿĿ����Ŀпѿѿѿѿѿѿѿѿ������#�!�����ݿѿ����������Ŀѿ�y���������������������y�n�m�e�g�m�o�y�y���
�
����������
������������������ûܻ���������лû��[�T�e�x����������(�4�A�K�M�Y�Q�M�A�4�(������������������	����	����������������������N�s�����������������������g�Q�N�G�F�G�N�����¾ʾ׾��������׾ʾ��������������������������������y�w�o�y�������������������ûŻû�����������������������������¦­©¦�y�u�t�q�t�v�����
����������������������������������������������������������������޺'�+�3�7�3�0�,�'��������#�'�'�'�'���������¼ʼ̼μʼ�����������������������#�0�3�5�0�0�#������������à������������óÓ�z�m�a�Y�U�W�o�o�zÇà�����������������������������������������s�����������s�f�c�f�l�s�s�s�s�s�s�s�s����������������������)�5�B�M�J�B�5�)����������������������ĺϺֺֺɺ��������z�m�l�p�|��DoD{D�D�D�D�D�D�D�D�D�D�D{DqD]DVD^D\DbDo�������ʼμҼ���ּʼ�����������������EiEuE�E�E�E�E�E�EuEtEiEdEiEiEiEiEiEiEiEi , < X ] J r Q j # W q U m 6 8 - + ] ' _ V N ! f B ? E I S r b ; : M U j 3 J Q d 6 ] ) � L C   6 F 4 l : [ m 4 Q % B \ ` > [ < 1 o X  �  <  �  �  t  7    Q  �  a  �  .  �    R  �    x  �  �    �  p  �  �    �  !         =  �  �  �    �  �  �  s  f  m  
  �  �  V  �    �  �  ^  
  F  �  �  �  �  �  <  K    �  o    0  �<D���ě�;�o�D��:�o��o:�o;o=��<o;�`B=�1;��
<e`B=@�<t�<��
<���<D��<T��<�C�=]/=C�=0 �=��=+=o<��
=�1=��<ě�=��P=<j=#�
=8Q�=C�=�+=�=T��=�P=��=T��=H�9=�w>�R=H�9=�+=�;d=L��='�='�=L��=<j=D��=P�`=aG�=L��>hs=]/=e`B=e`B=��=�
=>Kƨ>��>I�B"�[B"��B6�B �sB��B 1�B �B�B\'B ��B�BB��B��B&
�B�^B9�B	�B�cB5dB
�Bq�A�wB�B��B!S�B��B
�>B2B��B�hB��B![�B%�BE�B��B�B��B�)BBU�A��-B�ZB	B3�B�B# hB�1B
��BB,�RBr�B��B�TB�2B�WB"��A�K�B�B0+B�B';3B��B(B0�Bl�BB�B"�B"�B@B �iBȮB F�B 7PB�uB,�B �WB�%BUpB�B&?�B�B?�B	�?B�@B?�B
�,BB;A��ZB�\B�5B!>;B��B
�IB��B;�B$B��B!�|B;�B��B�B�BÌBE�B��BG�B >BQ�B	:@B�B��B#&B��B1�B@B,z�B��B�'B�B47B�B"@wA�:zB>"B?�B>�B'<wBˆBļB?�B��B@@@���@"�~?���?��?G�B>�Z�A�w]A��nA��A��mA�5�A��`A�lU@��Ahz!A$��A�yA?�A}|VA��B !�A�$�Be�B	d�A]zA>��B #�A�;�A�BA�}�A}�A�HA�wA�n�A�a*A΄tA�[{@�E�A�z1A�S�Aza�A�Am��A��@���A7��A��8A��+AP��A.�@��A�~,?35�A�h(?���@�kOA�F�A�
9AKy]AD}�A���A��@�PC�и@�xfC��@�J8@#�-?�X�?��?P�%>��8A��jA��A�aA�}A݃�A�A��Z@��0Ag�A# 8Aр A?5�A}^?A��OB @xA�{B:�B��A\��A>��A��PA��8A�{�A��mA|ȘA�z�A�y'A�|�A�|�A��A�~C@��A�~�A��)Az�?A�AmLA��q@��A6HOA���A��AR��A�_@�	3A��r?D:�A�s?���@��A�)�Aʃ>AK�AC�A�f�A��@	�C���@��C��~   %   	         
            A         S         ,                  	   -                      C         7               +   \                     |      !   L                     	         ]            	   /   �   E      #                        =         I         '                     %      1   +            7   %               )         +                     3         +                              3                  #   #                              5         #                              !      1               5   %                                                                                                            N�dN ��N��'N."kN\M�s7M�<�N,?P�`�N*i�N?j�O�'
NbRN�!�Ow��N�XkO A	NĶ�N���N4��N�ˣO�kO>�LP*�nOc��N�lO,tM��Pj�QO�0�NǦO��$O�SN5�[Oq��N��N��O��O2A�Nk,YN\�OeO	N��N�@hO+�O�O'b�O���Oc�NC�N�UN�*�N8�N��uN�^�N��N��O�؟NɣNB��M�z�N�Y�O�ϏOK�ON9�Nk��  t  E  S       ]  �  �  6  �  Y  �  {  �  �  �  �  W  �  (    �  �  F  �     }  �    .  �  D  �  ,  {  3  �  
�  N  5  [  b  �    �  j  �  �    �  l  �  �  �  7  �  �  
�  �  (     *  �      I��o�e`B�49X�#�
�o��o��o%   <t�;o;o=<j;o;��
<���;�o;��
<49X;ě�<t�<t�<�t�<���<D��<�j<���<�C�<�t�<���<��
<��
<�1<���<���<��<ě�='�=��P<�h<��=o=C�=�P=o=���=+='�=u=\)=�P=�w=#�
='�=,1=0 �=0 �=49X=�Q�=L��=L��=]/=ix�=u=�x�=��>�����������������������������������������[[\hjt�������th[[[[[��������������������������������������������������������������������������������IN[`gigb[ZVNIIIIIIII)4HUn��������aUH.������������������������������������������)5BSVVNB5)�=EHUaanonaUH========#"#0<IKNII<0,#######��
(/4<?CB</#	���zw|����������������caghnt���������tigccmqtu����������ytmmmm��������������������~�������������~~~~~~��������������������"/;@HQTTRH;/("����$22/)$BJS[gtvlsmZN5��������������������)56<52)ssx{}{������������ts���������������������������)6@=6)�������������������������"#)6?BOYUOB6+)(������������������������������������������������������������#/<HNUYTRPH<.%���.,-/:<<HPUWVUKH<7/..MJKNOW[httz|wtph[OMMadht�������������tha��������������������mmqz~����zqmmmmmmmmm%%&*15BIMPUVRNB>3-)%ZSQZ[ggtuutjg[ZZZZZZknnz��������������nk�����������������������������   ��������$"#')5BNW[\[UNEB51)$gbdnt|������������tg������!+0/*������������������������V_`abnnswniaVVVVVVVV/,-05BDLNT[_[NMB85//���
	��������������������������������������������������������������"/;;==;/"������������������������

������������!#+/563/)#!!!!!!!!UUIGILU[bbbWUUUUUUUU�������������������� �
6BHJHB?;6) ������

��������������	�����������������������������������������������������������{�|�����������������������������������������������@�B�L�V�Y�_�b�Y�Y�U�L�@�=�:�8�@�@�@�@�@�'�)�3�7�6�4�3�0�'�!����$�'�'�'�'�'�'������������������������ùùϹ۹ܹܹܹϹϹù����ùùùùùùù�²¿��������¿²±±²²²²²²²²²²�������������������������������������������A�Z�������p�e�Z�N�5����߿ٿ׿���a�l�m�z���z�z�y�m�a�[�\�a�a�a�a�a�a�a�a�tāćċąĄā�~�w�t�t�o�t�t�t�t�t�t�t�t��#�I�U�e�n�x�~�{�n�b�U�;�,�%������a�m�o�y�m�m�e�a�_�[�X�X�a�a�a�a�a�a�a�a���������������r�r�i�j�r�~�������m�|�������y�u�m�e�`�T�G�;�3�3�8�;�G�`�m�������ĽɽνнннĽ������������������������������������������������������޾Z�f�o�s�y�x�s�g�f�Z�N�M�F�L�M�O�Z�Z�Z�Z�ѿݿ����������ݿԿѿɿɿѿѿѿѿѿ��;�>�H�M�H�H�=�;�;�6�/�-�/�7�;�;�;�;�;�;��*�6�;�C�O�O�\�O�C�6�*� ��������/�T�a�k�m�b�T�I�;�/�����������������/�O�\�h�uƁƎƎƏƎƉƁ�u�h�\�S�O�I�G�N�O�������D�I�V�m�w�o�V�0���������ƾ���̿	��"�-�:�<�7�1�.�"�������������	�Z�Z�f�g�i�g�f�Z�O�M�H�H�M�Y�Z�Z�Z�Z�Z�Z��*�6�C�O�\�d�X�O�M�C�6�*��������m�z��������z�m�g�i�m�m�m�m�m�m�m�m�m�m���	�/�=�F�S�U�T�H�;�"�	�������������������������������������s�[�R�Z�s���������˿ݿ��������������ݿܿ׿տѿпѿڿݿ�ù��������������ùìàÓÇÄÂÇËÓìùŠŭŹŹ��������������ŹŭŠşœŐŔŠŠ�6�@�B�B�C�B�6�)�&�'�)�2�6�6�6�6�6�6�6�6�	���"�"����	���������������������	����������ùøù������������������������������	����
������������������������'�4�@�A�M�Q�U�M�M�@�4�'�"��������#�(�.�9�<�B�B�<�;�/�#���
���	�����������������������������������������޿ѿѿݿ�ݿֿѿĿ����Ŀпѿѿѿѿѿѿѿѿ������!������ݿѿĿ������Ŀѿݿ�m�y���������������y�w�m�k�m�m�m�m�m�m�m���
�
����������
����������������ûлѻܻ���ܻлû�������������������(�4�A�K�M�Y�Q�M�A�4�(�������������������	�	���������������������������Z�g�s���������������������s�g�\�W�R�S�Z�����¾ʾ׾��������׾ʾ��������������y�����������������y�y�r�y�y�y�y�y�y�y�y�����ûŻû�����������������������������¦­©¦�y�u�t�q�t�v�����
����������������������������������������������������������������޺'�+�3�7�3�0�,�'��������#�'�'�'�'���������¼ʼ̼μʼ�����������������������#�0�3�5�0�0�#������������Óàìù������þðàÓÇ�z�o�j�l�uÇÊÓ�����������������������������������������s�����������s�f�c�f�l�s�s�s�s�s�s�s�s����������������������)�5�B�M�J�B�5�)����������������������úκպɺ����������{�n�m�q�|��D�D�D�D�D�D�D�D�D�D�D�D�D{DtDoDvD{D�D�D����ʼмּټ��߼ּʼ�������������������EiEuE�E�E�E�E�E�EuEtEiEdEiEiEiEiEiEiEiEi . < X ] J r Q j " W q K m 8 A - + % ' R V Q  f  G E I S r b : : J - j  ' Q d 6 X  � # C  ) F ? l : [ m 4 Q % D \ ` > [ ; + H X  �  <  �  �  t  7    Q    a  �  
  �  �  �  �    �  �  y      �  �  �  �  �  !        3  T  b  �    �  9  �  s  f  	  �  �  r  V  h  �  �  ^  ^  
  F  �  �  �  �  �  <  K    �  M  �  �  �  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  ?R  0  `  �  �  �    '  >  R  a  o  t  k  [  ?    �  �  i  t  E  Z  o  t  q  l  f  a  Z  S  K  A  6  +               S  E  5  #    �  �  �  �  �  �  �  p  V  <  &  �  I  �  �           �  �  �  �  �  �  r  N  '  �  �  �  �  S  $   �            �  �  �  �  �  �  �  �  u  O    �  �  5   �  ]  T  K  B  :  1  (      �  �  �  �  �  �  }  g  R  <  '  �  �     
      &  0  :  C  Y  z  �  �  �  �    @  a  �  �  �  �  �  �  �  �  �  �  �    z  t  o  j  e  `  [  V  Q  �    3  5  '    �  �  �  �  a  2    	  �  �  �  �    U  �  �  �  �  �  �    e  G  )  
  �  �  �  �  h  F  "   �   �  Y  Q  J  C  :  *    
  �  �  �  �  �  �  g  H  2      �  �  �  8  �  �  %  L  l  �  �  �  �  �  �  _  �  T  �  �    {  t  l  e  ]  V  N  @  .       �   �   �   �   �   �   v   ^   G  ~  �  �  �  �  �  �  �  �  �  z  d  N  4    �  �  d     �  �  N  u  �  �  �  �  �  �  �  a  6    �  �  8  �  b  �    �  �  �  �  �  �  �  �  �  {  r  i  _  S  H  <     �   �   �  �  �  �  �  �  �  �  �  x  \  :    �  �  z  3  �  �  %   �  Q  F  6  .  ?  F  L  Q  V  O  A  0      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  s    �  �  �        &      
  �  �  �  �  �  �  �  �  j  Q  8          	  �  �  �  �  �  �  �  �  u  X  ;    �       �  �  ;  {  �  �  �  z  E    �  �  �  �  �  l  	  �  �  f  �  �  �  �  �  �  �  �  �  �  �  �  �  �  g  1  �  �  }  (   �  F  =  &    �  �  �  �  �  f  @  (    �  �  �  �  X    �  k  w  ~  �  �  �  �  �  �  �  �  {  k  Z  G  4    �  �   �  �  �  �  �  �  �  �            �  �  q  "  �  l      �  }  Z  ;    �    �  �  �  �  �  �  d  =    �  r    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  g  �  �  �  s  �  �  e     �  9  �  �   �  .    �  �  �  �  t  K  #       �  �  �  �  b    �  �  }  �  �  �  �  �  �  x  g  V  E  6  )         �   �   �   �   �  D  =  -    
  �      �  �  �  z  @  �  �  �  !  �  �  �  �  �  �  �  �  �  �  �  �  s  Q  +  �  �  x  $  �  �  �  m  �  �  �  $  D  b  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �     =  I  R  [  j  w  y  p  b  P  7    �  �  j    �  �  3    �  �  �  �  �  n  Q  1  �  <  �  �  f  *  �  �  n  -  ?  �  �  %  G  ^  p  ~    p  M    �  �  ,  �  s    �  �  �  ;  �  	$  	Q  	�  	�  
F  
�  
�  
�  
�  
i  
  	�  �      �  x  N  >  )    �  �  �  �  �  �  �  t  @    �  ~  0  �  Q  �  5  %      �  �  �  �  �  �  �  �  �  j  H  &  �  �  �  F  [  W  S  O  J  E  A  =  :  7  5  4  3  4  9  =  H  l  �  �  U  Y  a  P  6    �  �  �  i  A    �  �  �  �  ^    �  �  y    �  �  �  �  �  �  }  a  B  !  �  �  �  �  r  J    �            �  �  �  �  �  �  �  �  y  _  =    �  �  i  
�  `  �  �  �  �  ]    
�  
�  �  �  �  �  
�  
,  	3  �  �  �  j  W  >  !     �  �    H    �  �  V    �  �  A    �  �  f  �  �  �  �  �  �  �  d  ?    �  �  K  �  Z  �  A  �  �  Z  �  �  �  �  �  �  �  �  �  �  k  �  p  �  �  2  ?  �  7        �  �  �  �  �  �  �  n  R  1    �  �  �  {  ?   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  S  l  `  T  H  <  /  #      �  �  �  �  �  �  �  �  �  v  g  �  �  �  �  �  �  k  M  .    �  �  �    T  &  �  �  �  d  �  �  �  p  [  O  B  5  )        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  Z  G  6  &      �  �  �  �  �  �  7  +        �  �  �  �  �  �  �  l  O  2    �  �  �  k  �  �  �  �  �  �  �  y  Y  9    �  �  �  �  �  o  -  �  �  �  �  �  �  �  �  �  �  �  �  �  }  l  [  J  8  (        	]  	w  	�  	�  
(  
p  
�  
�  
�  
�  
�  
�  
>  	�  	  >  2  �    �  �  �  �  �  �  �  �  �  �  �  d  ;    �  �  �  f  9     �  (      �  �  �  �  �  �  �  �  �  w  s  x  ~  ~  a  D  '     �  �  �  �  �  �  �  �  }  u  r  p  n  l  j  h  e  c  a  *    �  �  �  �  �  }  \  :    �  �  �  d  7  
  �  �  �  k  �  �  c  /  �  �  �  Y  "  �  �  x  C    �  ?  p  %   �  �    �  "  �  �  �    �  �  b    �    Q  M    	�  �  �  
}  
�  
�  
�  
�    
�  
�  
�  
4  	�  	_  �  ]  �  �  �  �  k  ?  I  C  =  3      �  �  �  �    a  D  &    �  �  F  �  6