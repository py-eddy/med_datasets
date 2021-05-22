CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��1&�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N{.   max       P�	�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��/   max       <u       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?h�\)   max       @F��\)     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vvfffff     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @P�           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �o   max       ��o       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�	]   max       B5g       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��'   max       B5>�       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��#   max       C�w�       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =О�   max       C�x}       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          g       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ;       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N{.   max       P��+       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���҈�   max       ?�IQ���       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��/   max       <D��       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?nz�G�   max       @F��\)     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �׮z�H    max       @vu\(�     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P�           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��            V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�}�H˓   max       ?�B�����     P  X�         !   X                                 (               %      3      &   X            %               f   J                           &         !         4         	      	      E            )            O<GN{.O�e�P VRN^�CN$��N�N�EIN�?�P
��O\o�N��N̕�O���P.�O�xOwz@NU<N���O�(xN��P�	�O}�(P��PN�,�OS��NH7O���O���O �N�NP:ZPBogO׻O��	N���O��N�"xO�?�N9�6N9LIO��P!��OUl�O
�'O���OyN���O�3)OV��O^�;N���O��N٭�O�O��~O4�OFY5N�+P�RNIݍN��!N�Q:N���<u:�o:�o%   %   %   ��o��o���
�#�
�#�
�#�
�49X�D���D���D���D���e`B�e`B�e`B�u�u�u��o��o��o���㼛�㼛�㼴9X��9X��9X�ě������������+�+�C��\)�\)��������w��w��w�#�
�''<j�D���L�ͽL�ͽL�ͽL�ͽL�ͽY��aG��m�h��������C����P��/���������������dgt���~xtoigdddddddd��
!##-3/#
�������
#(8@<74(
������;<HSU]ahfaUHE<;;;;;;��������������������)41-)&26<BFMORROGBB<663122��������������������JTaz������{~��zaTIFJ��! ��������������������������IO[_hrtrhh][YODFIIIIIOTamz������zmaTHEEITamz�������ngaTPQLJT������������������������������������~�����������������������������������������nv��������������{tqn8BN[a^_[RNB;88888888[g�����������teX\[oty�����������toljjo~����������������mg~���������������������������������������������������������������������������������
#/Hai`UJ</#
����������������������������������������������
#&+#
	����������������������������������������������������������t����������������zvt�����������������������
#05:,#
������������������������#*4<ITZb_]^lfI<0'""#����������������������������������������9<CIUbnv|yzxnkfbUD<9����������������).4:@5)�����������������������/468<GIUXZeill[UJ<0/�����
&/7=9/#
����
#++'#!
bjnz����������zna[Ybaenz�������zrna``^\a�����������������������
�����������%&-5BEDGQSQB5'))5BN[_[XONB51)())))]glq������������vtg]#/<HNTWXUSH/#	>BIO[ehntuxyvth[ODA>LOU[_hmtw��{xth[LIL��������������������Yat������������tcZXY���������������������$)+)' ������|�����������xw||||||[\hilhhg\[ONDCFOP[[[�{�{¦¬²µµ²®¦�s�p�r�m�s�~���������|�s�s�s�s�s�s�s�sÇ�z�u�x�n�`ÇÓìù��������������ìÓÇ�Z�A�6�*�%�(�5�A�I�g�s���������������s�Z�(�(�"�����(�5�8�7�7�5�*�(�(�(�(�(�(�"������"�&�+�*�"�"�"�"�"�"�"�"�"�"���
���)�-�)� ������������S�M�S�Z�_�l�p�x�������������y�x�l�_�S�S����������������������������������������ƧƐƂ�n�l�uƚƧ������������������Ƨ��������(�5�A�N�U�X�V�P�A�5�(���m�l�k�m�n�l�`�W�`�b�m�y�}����������y�m�"������"�.�;�;�G�H�G�D�;�.�"�"�"�"�5�(�"�������5�A�Z�g�o�|���s�Z�N�5�����������	�/�C�H�T�U�a�m�p�k�T�;�"�����T�M�T�V�a�h�m�z�}���������������z�m�a�T�����������������׾���������׾ʾ�����D�D�D�D�D�EEEEEEEEED�D�D�D�D�D�ùñìãâåìù����������ýùùùùùù�����x�w�r�o�r�x�����������ʻû����������	� �����	��"�'�"� ���	�	�	�	�	�	�	�	�	�������&�G�`�q�v�y�����y�m�T�"�	���������������ѿݿ����������ݿѿĿ������Ŀĳĭĺĵ��ľ���
�<�n�t�l�l�b�I��Y�;�'������'�@�Y�r�������������f�Y�U�T�H�?�<�5�/�<�H�U�U�\�a�b�a�W�U�U�U�U��������������������������*�,�*�"�ŭŧŠŗŠŭŹž��Źŭŭŭŭŭŭŭŭŭŭ��������)�B�N�[�h�g�Y�Y�O�6�)�������y�m�b�T�I�T�V�q�y�������������������	����������	��"�.�0�;�G�C�;�.�"��	�	�t�i�j�h�m�p�t�~�t�t�����$�0�6�4�0�$�����������黪���r�l�l�x�����л��
����	�����Y�M�4����%�@�M�f�r��������������r�Y���������������������)�5�;�=�4�)�������z�m�j�i�m�u�z�������������������������������������������������ĽȽнѽнĽ��������ʾξϾʾʾ������������������������������z�n�g�g�s�������������������������������ݿѿʿѿݿ����
����������������������������������������������Żܻлû��������ûлܻ���
��'�+�(���ܼY�R�W�f�v���ּ���!�����ּ�����f�Y�h�]�[�V�Z�[�i�tāčĚġĢĝĚđčā�t�h�V�N�L�N�O�J�V�W�b�o�s�{�~�{�z�v�o�b�V�V�M�(�������ӽ�����(�4�A�C�M�^�]�M�0�)�$�"������$�0�=�F�L�N�O�L�I�=�0�����������������������������������������ù����������������ùϹܺ�� ��	�������	� ��	��"�/�;�H�O�Q�T�X�]�T�H�;�/��r�g�i�X�]�_�b�e�v�~�����������������v�rÇÀ�zÂÇÓàìõìèàÚÓÇÇÇÇÇÇìàÓ�z�n�h�i�nÇàù��������������ùì����������������������� ��������������������ŹŭŠśŞŠŭŹ��������������������E�E�E�E�E�E�E�E�E�E�FFFFFFE�E�E�Eٽ����������������½Ľнݽ�����ݽн��ù����������������ùϹܹ���������ܹϹ��|�z ¦²¿��������¿²ĳĦĜėĖęĦĳĿ��������
�
�����Ŀĳ�������������������������������5�.�)�0�5�=�A�K�N�R�Q�N�I�A�5�5�5�5�5�5�#����#�/�<�<�H�M�H�E�<�/�#�#�#�#�#�#�g�g�q�s�����������������������v�s�g�g�g  U 0 : g R < g X a 2 W 9 N D F ^ b < ; F 9 4 R 3 R n a S H J ` 8 . 5 b & S A 7 � G . u B E [ L : N 6 Z C b ? / $ . $ � & + < $ 8    9  M  �  �  �  b  >  �  �  �  �  �  �  �  �  f  2  �  �  e  �  �  �  �  �  �  -  D  �  @  6  7  d  t  �  a  �  e  �  M  �  T  g  �  �  V  J  $  �  g  �  8  �  �    6  �  �  �  z  Y  Z  �  �  �ě���o��󶽲-��o��o�ě���C��t��C���j�T����o��P�P�`��`B�������/�L�ͼ�1����#�
�]/������`B�t����ͽaG��\)�C��C���h�o���ͽ0 Ž�w�<j�,1�<j��P�8Q�aG�������C��H�9��t���7L�8Q콾vɽ�%����q����%�q�����P�����+���罅���
=��\)���E����#B]B	��B�B�2B�aB�NB�oB![B�A���BQB ��BL4A�	]A���BM�B��Bm1B�VB̃B-[BǲB
�@B�{B �bB!'�B��B�PBXB��B<�B�wB�OB�3BRB�)B��B$�VB5gB&��B)�dBZB'aB-�B5�BzRB&ɮB��BF�Bz�B�B!'B��BY,B��B
�BOsBU�B��BdB
��B"�PBf7B
�'B_BF�B
1:B+�Bl�B8�BB��B?�B?TA��'B?�B!7bB;hA�zbA���B@BB A�BAB��B�ABR�B
ǞB
��B��B ��B!/B{B�^B�hB��B@�BE�B�NB��B@�B
��B�B$�WB5>�B&LB)�B�nB'<@B-B�B
7B��B&��B�\B4xB��B
^B!@�B�B�B��B
�B�yB@BRB�B
C�B"��BS�B
��B<�A�v�ADEA˺)A�s/A��/A�3cA�J�@���A�j�BФA��YAlrA`��A�̖A�4A�*AO�C�SA�+�@��A��Ac4IA|A��@�A�Aħ�A���A���A�H�An��A^s�A��jB	˵@�L.@�G�A���A��A$��AN4.A�3�A�]A�Ժ@��R@���A�P�B3[A6]�B
X�A��>��#A�:p@,�Aʯ�A�"KA���A�!)C�w�A(bh>���A��A�y@Y�bA�7�A¹�A��A�~�ACA�l�A���A���A��A���@���AЕ�B�A�^�Al�_A`�eA�xA�sFA�N�AP\C�UWA�w�@��A��sAe�fA|��A���@�`�AăxA���A�|*Aה�An��A]��A���B	�I@�wh@ܨ=A��0A�F�A"��AN�!A��A�A�{�@�m�A�&A���B��A6�B
@NA��A=О�A�Q@	WA�mA� A���A��C�x}A(ʶ>��A�nA�s�@Wl�A��A�@A�p�         !   X                                 (               %      3      '   Y            %               g   K            	               &         "         5         
      
      E            )                     #   )                  -            !   '      %               ?      9   -            !               1   #                        !   3         !         '                                 %                                          %                     %               ;      9                              '   !                           3                  #                                 %            N�)N{.Om��O��8N^�CN$��N�N��`N�?�O��O5	N��N̕�O���O��Nr�vOwz@NU<N���OK=�N��P��+OR�xP��OXNHA�OS��NH7O��O���O �N�NP:ZP�O���O��	N���O��N�"xO�?�N9�6N9LIOcTP!��OUl�N��%O;�EOjO�N���O�uO3�[O N���O��N٭�Ov4O�,dO!QO�N�+O�*8NIݍN��!N�6N���  j  �  �  	�  �  ;  �  �  �  :    �  t  S  �    �  y  �  r  f  �    �  �  �  �  �  �  �       {  
z  
�    �    �  %  �  �  �  �  �    �  7  z  B  I  6  R  �  #  �  �  Q  2  �  l  �  �  t  f<D��:�o���
�+%   %   ��o���
���
�e`B�T���#�
�49X��o��/����D���e`B�e`B��j�u��C���t���o��o���㼛�㼛���/��9X��9X��9X�ě��D����P���+�+�C��\)�\)���,1����w�#�
�49X�''0 ŽD���T���L�ͽL�ͽL�ͽT���m�h�]/�u�m�h��7L�����C�������/���������������dgt���~xtoigdddddddd����
'+-,#
���������
"%%# 
�������;<HSU]ahfaUHE<;;;;;;��������������������)41-)&36;BELOQPOB=66313333��������������������GMTamz�����|uurmaTIG�����������������������������IO[_hrtrhh][YODFIIIITamz����}zmaTNJHHJMTT[amz�������|mfa\WTT������������������������������������~�����������������������������������������}����������������zw}8BN[a^_[RNB;88888888^\gt������
 �����gY^rt�����������wtromnr~����������������mg~���������������������������������������������������������������������������������
#/HVYUOC</#
��������������������������������������������
#&+#
	���������������������������������������������������������t����������������zvt�����������������������
#05:,#
������������������������#*4<ITZb_]^lfI<0'""#����������������������������������������IUbhoqqkhfbUJI@?>>AI����������������).4:@5)�����������������������:<GIUV\aegeUIF<3368:���
%/6<8/)#
�����
#++'#!
cknu����������zna\Zc`agnz��������znda`]`�����������������������
�����������%&-5BEDGQSQB5'))5BN[_[XONB51)())))nt�������������ytjln
#/<HNQUULH</#
N[dhlstwxuth[XOFCCDNNOX[hqt}tutohc[XOLN��������������������Zct������������td[YZ���������������������$)+)' ������}�����������yx}}}}}}[\hilhhg\[ONDCFOP[[[�~�~¦§²²«¦�s�p�r�m�s�~���������|�s�s�s�s�s�s�s�sÓÇ�z�z�yÀÇÓàìù��������ùðìàÓ�Z�N�D�?�?�N�Z�g�s�������������������s�Z�(�(�"�����(�5�8�7�7�5�*�(�(�(�(�(�(�"������"�&�+�*�"�"�"�"�"�"�"�"�"�"���
���)�-�)� ������������S�N�S�\�_�l�r�x���������x�w�l�_�S�S�S�S������������������������������������������ƧƖƈ�}�x�}ƚƧƳ������������������������(�1�5�7�A�J�K�D�A�5�(����m�l�k�m�n�l�`�W�`�b�m�y�}����������y�m�"������"�.�;�;�G�H�G�D�;�.�"�"�"�"�'�����(�5�A�Z�`�j�u�|�s�g�Z�N�A�5�'�	�������������"�/�;�C�H�J�H�D�;�/�"�	�z�v�m�j�m�w�z���������������z�z�z�z�z�z�����������������׾���������׾ʾ�����D�D�D�D�D�EEEEEEEEED�D�D�D�D�D�ùñìãâåìù����������ýùùùùùù�����~�|�x�x�{�������������û������������	� �����	��"�'�"� ���	�	�	�	�	�	�	�	�"�	����������(�G�`�p�t�s�w���m�T�"�������������ѿݿ�����������ݿѿĿ������Ŀĳĭĺĵ��ľ���
�<�n�t�l�l�b�I��M�I�@�9�8�@�E�M�Y�f�r�v�������r�f�Y�M�H�A�<�9�7�<�D�H�U�Y�^�U�H�H�H�H�H�H�H�H��������������������������*�,�*�"�ŭŧŠŗŠŭŹž��Źŭŭŭŭŭŭŭŭŭŭ�����!�%�)�6�B�O�[�b�_�T�R�O�C�6�)������y�m�b�T�I�T�V�q�y�������������������	����������	��"�.�0�;�G�C�;�.�"��	�	�t�i�j�h�m�p�t�~�t�t�����$�0�6�4�0�$�����������л��������������ûл�����������ܻмf�Y�M�4�*�#�-�@�M�Y�f�r������������r�f���������������������)�5�;�=�4�)�������z�m�j�i�m�u�z�������������������������������������������������ĽȽнѽнĽ��������ʾξϾʾʾ������������������������������z�n�g�g�s�������������������������������ݿѿʿѿݿ����
����������������������������������������������Żû����ûлܻ�����"�"�������ܻлüY�R�W�f�v���ּ���!�����ּ�����f�Y�h�]�[�V�Z�[�i�tāčĚġĢĝĚđčā�t�h�V�R�N�O�Q�V�b�o�r�{�}�{�x�t�o�b�V�V�V�V�������������(�4�@�K�M�X�M�A�(���0�$������$�0�=�E�I�L�N�N�M�K�I�=�0�����������������������������������������ù����������������ùϹܹ�����������"��	���	���"�/�;�H�M�N�S�H�B�;�/�"�r�e�a�d�e�f�|�~���������������������~�rÇÀ�zÂÇÓàìõìèàÚÓÇÇÇÇÇÇìàÓ�z�n�h�i�nÇàù��������������ùì����������������������� ����������������ŹŭŢŠŝŠŠŭŹ��������������������ŹE�E�E�E�E�E�E�E�E�E�FFFFFFE�E�E�Eٽ����������ĽŽнݽ������ݽнĽ����ù¹����������ùϹܹ�����������ٹϹ��|�z ¦²¿��������¿²ĳĦĝĘėĜĦĳĿ���������������Ŀĳ�������������������������������5�.�)�0�5�=�A�K�N�R�Q�N�I�A�5�5�5�5�5�5�#���#�#�/�8�<�H�L�H�D�<�/�#�#�#�#�#�#�g�g�q�s�����������������������v�s�g�g�g  U % 4 g R < l X Z ) W 9 F * $ ^ b < * F : 1 R - V n a T H J ` 8 ' / b & S A 7 � G < u B 7 E K : N . = C b ? !  + % � # + < # 8      M  �  C  �  b  >  �  �  3  @  �  �  J  �  v  2  �  �  �  �  �  �  �  M  a  -  D  K  @  6  7  d  m  r  a  �  e  �  M  �  T  �  �  �    �  �  �  9  w  Z  �  �    �  F  V  4  z  .  Z  �  �  �  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  V  `  h  h  a  V  I  5  !  
  �  �  �  }  M    �  `    �  �    {  v  r  n  j  f  a  ]  c  q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  Y  9    �  �  �  '  �  �  7  �  f  �  ^  �  �  	O  	�  	�  	�  	�  	�  	t  	-  �  _  �    >  �  �  u  �  �  �  �  �  �  �  �  �  �  �  �  w  _  H  /     �   �   �  ;  7  4  1  .  *  &  !          	    �  �  �  �  �  �  �  �  ~  z  v  r  n  k  g  c  a  a  b  b  b  b  c  c  c  c  �  �  �  �  �  �  �  o  X  >    �  �  �  M    �  �  U    �  �  �  �  �  �  �  �  �  �  �  �  �  y  q  �  �  �  �    1  7  0  7  7  /  )       �  �  �  �  T    �  �  D  �  r  �  �  �      
    �  �  �  �  �  �  l  @    �  o     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  a  M  9  &    t  o  k  f  b  ^  Y  U  P  K  F  A  7  )       �   �   �   �  7  ;  <  Q  L  ?  /      �  �  �  �  �  [    �  g  �  �  �  /  Z  s  �  �  �  �  �  �  `  3  �  �  Z    �  5  �  �  �  �  �  �  �  �          �  �  �  �  V    �  �  T  �  �  p  `  T  R  P  N  K  G  D  @  =  =  <  :  7  5  -  #    y  v  i  Z  H  4       �  �  �  S  $  �  �  �  h  *  �  N  �  �  �  q  Y  ?  #    �  �  �  }  Y  2    �  u  A    �  �  E  _  k  p  p  d  R  7    �  �  ~  7  �  �  D  �  �  �  f  _  X  R  Y  b  l  j  a  X  Q  L  F  E  N  V  a  |  �  �  u  �  u  ^  =    �  �  �  \  $  �  ]  �  �  �  O  �  g  �               	  �  �  �  �  q  ;  �  �  ]  
  �  &  �  �  �  �  �  �  �  �  �  �  �  v  a  B    �  u  	  �    �  �          6  K  c  �  �  �  �  �  [  "  �  S  k  V  l  �  �  �  �  �  �  �  �  �  �  �  m  I  %    �  �  �  �  �  �  �  �  �  x  [  7    �  �  �  h  ;    �  �  e    �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  Z  B  *    �  �  �  �  �  �  �  �  �  �  �  �  �  \     �  {  /  �  >  �  �  1  �  �  �  �  �  �  y  d  L  3        	  �  �  �  �  ]  '        �  �  �  �  �  �  t  O  "  �  �  �  d  4    �  �     �  �  �  v  O  '  &  :  V  `  [  e  e  N  %  �  �  Z    {  i  X  G  6  %      �  �  �  �  �  �  �  �  �  �  x  h  	�  
+  
W  
q  
z  
v  
g  
L  
*  	�  	�  	�  	%  �  .  �  �  �  v   �  
6  
v  
�  
�  
  
e  
?  
  	�  	~  	%  �  [  �  b  �  9  B  �  Z        �  �  �  �  �  �  �  �  �  a  @    �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  p  K  &    �  �  �      �  �  �  �  �  �  �  �  �  h  J  $  �  �  �  G     �  �  �  �    t  h  X  H  6  #    �  �  �  �  �  E     �   �  %          �  �  �  �  �  �  �  �  }  h  V  B  ,     �  �  �  |  v  p  k  e  `  Z  T  O  J  E  @  ;  6  1  -  (  #  �  �  �        )  5  A  >  8  1  "      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  9  �  �  m  H    �  �  �  �  �  e  ;    �  �  >  �  �  �  Z    �  ,  ~    �  ~  q  _  I  2    �  �  �  X    �  �  M  �  �  K  �  ?  �      �  �  �  �  �  �  |  \  8    �  �  �  }  p  A    F  {  �  �  �  �  �  �  {  d  F    �  �  k  �  h  �  5  �  /  4  '      �  �  �  }  C    �  �  ?  �  �    h  �   �  z  r  j  a  Y  P  B  4  &        �  �  �  �  �  �  �  �  '  B  ?  8  )    �  �  �  �    P    �  f    �    Q    E  F  H  @  4  )        �  �  �  n  ?    �  �  �  R        �  3  5  *    �  �  �  J    �  �  �  *  1    
  �  R  J  B  :  3  '    
  �  �  �  �  �  �  �  l  M  -    �  �  �  �  �  �  �  �  �  �  �  �  |  h  Q  8    �  �  g   �  #      �  �  �  �  �  �  i  J  )    �  �  �  �  �  �  �  �  �  �  �  �  �  t  U  3    �  �  �  �  T    �  �  b  ;  �  �  �  �  �  �  V  �  �  +  
�  
B  	�  	  O  \  L    a  �  J  O  N  H  <  *    �  �  �  �  {  V  /    �  �  Z  %   �    '  0  /  2  '      �  �  �  �  �  �  �  }  Z  0  1  X  �  �  k  \  s  �  �  t  _  C  &  	  �  �  �  {  W  "   �   �  [  l  _  >    �  �  }  W  -  �  �  {  7  �  �  E  �  _  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  Z  B  �  �  �  u  R  5    �  �  �  z  M    �  �  j  (  �  �  �  e  q  j  Z  E  ,    �  �  �  z  J    �  �  Z    �  �  Y  f  E     �  �  �  p  B    �  �  v  >    �  i    �     �