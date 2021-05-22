CDF       
      obs    @   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�vȴ9X        �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�T�   max       P9��        �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���-   max       <ě�        �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>p��
=q   max       @F�ffffg     
    �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���G�{    max       @v�p��
>     
   *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @O            �  4�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @�u             5,   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���   max       <T��        6,   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�M�   max       B5>%        7,   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��4   max       B4�9        8,   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >Wh�   max       C��{        9,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       > `c   max       C���        :,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          x        ;,   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7        <,   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5        =,   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�T�   max       P#�/        >,   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�!-w1�   max       ?�4֡a��        ?,   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       <ě�        @,   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>p��
=q   max       @F�ffffg     
   A,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ٙ����    max       @v~�G�{     
   K,   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @O            �  U,   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�&        max       @� `            U�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�        V�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���@��   max       ?�333334        W�         	                              #         .            ,      	      "      G      w      "                     +      '      (         M                           D         :               W         N�$�N���NǡNN�,N��N!}�O�9P,&\ON`N��O���N��Pm�O>(O*)�P/��OM�nOTԆO(5�O��O���O�O��_O�gO�"IP#�Nđ�O���N��1O� N�J�O��N�ѱOl�O+
gN�PO���Nt��P;�N��O���OKw�NN�EP9��O�E%N��N�߾N���N��	N��ORY�M�T�P^iO2��Nm��O�L�O�4N���OT�uN(�O���O&�O�@OT�<ě�<���<�o<u;��
;��
;D����o��`B�o�o�t��e`B�e`B�e`B��t����㼛�㼛�㼬1��j��j���ͼ��ͼ��ͼ�h��h���o�C��\)��P����������������w��w��w��w�0 Ž8Q�8Q�<j�@��@��@��H�9�H�9�L�ͽL�ͽP�`�]/�aG��u�y�#�y�#��o������P���-)6862*)$����� 		������������xz}�����������zyxxxCHKTYafaUTHFCCCCCCCC��������������������!#0<<?<0#!!!!!!!!!!�������)05/)����(<Unz�������naWK/##(?BN[ghhptqga[SNFCB>?���������� ��������6ABM[hmnfg[OB:551-/6>HUW[ada^UUHGD@>>>>>������������������������������������<BNOX[`bhjjhf[OB>89<x�����������������wx
#08<D<940#	
1<Ib{��������{hbIF:1��������������������jqz�������������zlij5BNg�������tgB5-%�������������������������� ����������%<HUagnvwnbtoUH<:-#%��������������������Zaz����������zshfaVZst}���������|utmssss	 �#/;ELQSRH</#	�������	
�������jz���������������zmj)0+)����������������������������������������st{��������������{ns��������������������<<DRU^mnsvronbWUI<:<bn{���������{nhb[WZbz{������������{yzzzz�����������������������������������s~����������������ts)6BO[]YOIB4)���������������������5[[WU]^[NB5���������������������ENZ[^_[NMGEEEEEEEEEEJOSZ[fhotwtqh[OLJJJJGHPUYanvypnaXUOHGGGG����������������������� ���������������4<ITZ^aa^UIGC<752014NUXbcbbUSKNNNNNNNNNN���-36,)������))/58==5/)

")-5@>5)�������
������STamz~zsmldaXTLIJKKS��������������������goty������������tnfg�����������������������
#/6><E<#
�����),)'������#)bgit�����������tsgcb�������������������������ֺϺ̺Ӻֺٺ��������������ùíìéëìù��������������ùùùùùù��üùôöù����������������������������ā�x�t�p�t�āĄččĎčāāāāāāāā���������������������������	���������������������¿¦¯²¿������� ��������/�#�!�"�%�;�T�a�m���������������r�H�;�/�A�<�.�(�2�4�A�Z�f�s�}�����}�s�f�Z�M�A���	�������������������	����'�"����	����.�;�G�`�m�y�����y�m�`�T�;�.������������������������������������g�_�]�[�T�W�Z�g�s�������������������s�g���������������������������������������������x��������������ξ־Ѿʾ��������r�d�Y�V�_�r���ɺ����/�:����ɺ����ۻлû������ûѻܻ������������ۻ-�'�$�%�)�-�:�F�S�_�g�l�q�g�S�K�J�F�:�-�y�m�c�`�\�]�`�m�y���������������������y�I�<�0�(�%�&�/�<�H�b�{ŔŚŔŊ��q�b�U�I�׾ǾľȾ۾ܾ����	��� �"����	��׾׾Ѿ;ʾȾʾξ׾��������������׾������������������*�6�?�K�O�O�C�*������������������#�2�<�H�U�U�O�L�H�#�
���@�;�=�6�8�4�6�@�L�e���������������~�e�@�������u���������Ϲܹ�������ܹɹ�������¿»¿¿����������������������������F$FE�E�E�E�E�E�E�FFF$F=FJFPFYFWFLF1F$�H�;�/�"�����	��"�/�;�F�H�T�_�T�Q�HƳƦƟƒƏƚƳ��������������������Ƴ����������������������� ����������������Y�W�N�M�I�M�Q�Y�f�r�~�����������r�g�Y�s�g�k�s�~�����������������������s�s�s�s�����x�p�l�j�q�x��������������������������ܻĻͻܻ������'�4�7�4�,��������g�f�Z�N�K�A�5�4�5�A�N�Z�\�f�s�t�z�v�s�g��ܻ߻� ��=�Y�e�`�d�n�r�n�f�U�4��������������������'�(�0�(�����������r�f�r������Ӽ���������ʼ��T�Q�G�;�2�.�)�'�.�;�=�G�T�W�`�f�d�`�T�T�������������������������&�*�*�(�����������������������������������������������������������������������������������������u�p�p�n�y�����������������������[�B�6�1�.�5�1�5�O�[�eāčĖĒčā�w�h�[�5�/�5�6�B�N�T�P�N�B�5�5�5�5�5�5�5�5�5�5�ù����������������ùȹϹڹܹӹϹùùù��g�e�Z�Q�N�C�C�N�Z�g�g�o�s�t�s�m�g�g�g�g�����������!�+�&�!����������ɺĺźɺֺ޺���ֺɺɺɺɺɺɺɺɺɺɽ��������Ľнݽ����$�������ݽнĽ��4�/�4�?�A�M�N�O�M�A�4�4�4�4�4�4�4�4�4�4ùìÛØàìù�����6�B�E�<�0�)�����ù�h�^�O�F�O�O�[�d�h�tāćčĒěĚčā�t�h�I�>�=�<�=�A�I�V�Z�[�V�V�I�I�I�I�I�I�I�I�4�'�� ������� �4�@�Y�r�~���}�w�f�@�4�����������&�(�5�:�A�G�A�5�(���ŇŅ�{�n�b�W�U�R�U�b�n�w�{ŇŐňŇŇŇŇ�$�����������$�)�0�J�J�B�=�0�$��	����������!�����������EED�D�D�D�EEE*E7ECEEEMELEGE=E7E*EE�!�.�3�:�G�M�X�Y�V�S�G�:�.�!������!ĦěĚĕĘĚĢĦĳĿĿĿ����������ľĳĦ�����������������ĿѿӿҿӿѿϿĿ������� M $ [ : | L J 9 ; J ; H 9 D B g ' { * 0 I = / - 8 = 2 R w c I * ` J i x : o V d % @ I D G I " V H R R D O ; M 1 V g K U * = - J  �  �     G  k  J  #    �    c  �  �  �  �  �  �  p  f  �  l  0  a  �  �  �  �  �  m  ?  �  >  �  T  �  c  \  �  o  �  �  �  h  r  M  J  �  �  �  $  �    (  ~  �  +  l  �  �  T  s  m  K  ;<T��<t�;�`B<D��:�o;D�������o����49X���ͼ�o�H�9������%�49X���\)����P�`���0 Žq���T�������o��u�#�
��7L�,1�Y��,1�]/����'��
�,1���-�8Q콡���L�ͽ8Q��F��hs�@���7L�T���T���L�ͽ�O߽Y������}��`B��+�}󶽟�w���P����\��9X���B�4B��B S]A�[�B�B%��Bn.B�B��B��B�By�BnBNB�{B��B%4�B'�gB*��B �B��B��B Bv�B!�KB��B�B�A�M�B�%B��B ��BgeB�B _B'LhB(�LB)VNB,��B��B��B�lB5>%B�B]�B^B\�B%�B#0�B#i0B&�SB'UDB��Be�BľB�A���BB�B
��B��BBG�B
#�BdeB�'B�B AZA�W�BH4B%�PBSeB�B��B��BH�B��BBBqB��B��B%>B(>'B*��B ��B	4�B�B7�B�/B"-]B��B��B�A��4B�BąB p�B��B,+B�yB'O>B(��B)�zB-?bB�sB��BќB4�9B�B0�BK�BK�B?/B#>B#^�B&��B'D�B��BR�B��B��A�| Bs�B
F�B�wB�jBA�B
/B>y@E�-A���A�b'A݄�?m��A
y�A�0A��A?6	A��QAeU�A��A�8EA���AL�&@,W@��@�v�Am�kA�%oAX�2AT�,A�c/A���?�:>q��A���C��{A��Bh�A�+<@�[�A��i@�$�@���A���@�<�A���A vAd�A�&A� #AK8IA�D-A���A�n@>Wh�A��C@`� @<eSA,6A:�A�9pA�g�Bhz@яA���A��B	܈BΩC�x�A��A��Awie@E��A�z�A�pUA�h}?k!TA
�}A�
A�}�A@�A�5=Ac.9A�~�A���A�[AL�P@�"@��S@{�Am�A�UAY�HAU�A��lA��Z?ׅ> `cA�(RC���A�]�B]-A�}@�T�A��+@�ʓ@��A��@��8A���A�Ad��A�x�A��
AK+�A��mA�NA�&�>D��A��	@`2<@9NA*��A:�AЉ�Aܚ�BB@��5A��6A�`�B	�LBA�C�|-A�%A�@Aw�         
                               $         .            -      	      #      G      x   	   "                     +      (      )         M                           D         ;               X                              #   -               %         7            !   #         !   !   +      "      '                     '      3      #         -                           +         '                                                               !         5            !                  !                                       -               %                           '         #                        N�$�N�ϻN�0�N�,N��N!}�O��O��N�l�N��O�)�N���O�O�ZN��GP#�/O2��OTԆOjO��O�ZO�O��_OY��O�ĹO۬Nđ�OgA�N��1N�܅N^�O��N�ѱN�_�O+
gN�PO��Nt��P
�N��O�eO&UCNN�EPYJO�E%N��N=�bN���N��	N��ORY�M�T�P(�O)YNm��O�QO�4N���O.	1N(�Os�BOj�O�@OT�  S  2  <  h  �  �  �  �  �  �      E  t  S  h  �  �  ~  �  �  �  �    �  	O  �  �  �  O  �  \  s  �  �  �  �    :  �  �  �  �  
*  n  U  �  �  �  �  j  	  	!    <  �  �  �  �  W  �  �  �  E<ě�<�t�<e`B<u;��
;��
�49X�D���49X�o�#�
�#�
��j��t����㼣�
��1���㼬1��1����j���ͽ\)���H�9��h����o�H�9�t���P���'����L�ͽ��'�w�8Q�#�
��w�e`B�8Q�8Q�Y��@��@��@��H�9�H�9�ixսP�`�P�`�q���aG��u��%�y�#�������P���P���-)6862*)$������������������zz�������|zyzzzzzzzzCHKTYafaUTHFCCCCCCCC��������������������!#0<<?<0#!!!!!!!!!!���� �����./<HUnz����znaUD<3-.NN[_fgjmjg[NNIHDNNNN���������� ��������6BE[cilidd[OB<673/16AHUVZaba\UHFB@AAAAAA����������������������������������������@BJO[]dca[POLB?=@@@@�����������������{{�	#05:<:710#
	1<Ib{��������{hbIF:1��������������������jqz�������������zlij'/5BN[gt����tgB5-(%'�������������������������� ����������,/?HUadinoga[UHA<2-,��������������������ioz�����������zupmfist}���������|utmssss#)/3=CFFB</#
�������	
���������������������������
)-)'
����������������������������������������stv}����������tssss��������������������<<DRU^mnsvronbWUI<:<bdn{��������{ngb`^^bz{������������{yzzzz����������������������������������}����������������{z})6BO[\YWOGB96) ��������������������)BIMPY[XNB5&��������������������ENZ[^_[NMGEEEEEEEEEENOY[hhqkh[RONNNNNNNNGHPUYanvypnaXUOHGGGG����������������������� ���������������4<ITZ^aa^UIGC<752014NUXbcbbUSKNNNNNNNNNN���)/21)!������&).57<<5.)")-5@>5)�������������STamz~zsmldaXTLIJKKS��������������������t{�������������tpiit����������������������
'-24/)#
����� "))+)&��� bgit�����������tsgcb�������������������������ֺϺ̺Ӻֺٺ����������������üùìêììù����������������������������ùü��������������������������������ā�x�t�p�t�āĄččĎčāāāāāāāā���������������������������	���������������¿¸²¦¢¦¨²¾¿������������������¿�;�9�.�+�+�0�8�H�T�m�z�������z�m�c�T�H�;�A�A�9�A�M�Q�Z�f�s�v�{�t�s�f�Z�M�A�A�A�A���	�������������������	����'�"���!���%�.�;�G�T�`�m�u�}��y�m�`�T�;�.�!�������������������������������������s�g�d�c�`�a�g�s�����������������������s�����������������������������������꾥�����������������žʾ˾ʾƾ��������������f�[�X�b�r�����ɺ���%�0�����ɺ�������ܻлû������ûлܻ������������-�'�$�%�)�-�:�F�S�_�g�l�q�g�S�K�J�F�:�-���y�m�e�`�]�_�`�m�y���������������������I�<�0�(�%�&�/�<�H�b�{ŔŚŔŊ��q�b�U�I��׾Ҿ;оܾ�����	������	�����׾Ѿ;ʾȾʾξ׾��������������׾������������������*�6�?�K�O�O�C�*���
������������
��#�/�5�<�G�C�<�5�#��
�L�@�;�=�8�@�L�Y�e�r�~�������������~�e�L�������������������Ϲܹ���	����ܹ�������¿»¿¿����������������������������E�E�E�E�E�FFF$F1F=FGFIF@F1F$FFE�E�E��H�;�/�"�����	��"�/�;�F�H�T�_�T�Q�H�������������������	���������������������������������������������������������Y�W�N�M�I�M�Q�Y�f�r�~�����������r�g�Y�s�g�k�s�~�����������������������s�s�s�s�������x�s�m�t�x��������������������������ܻĻͻܻ������'�4�7�4�,��������g�f�Z�N�K�A�5�4�5�A�N�Z�\�f�s�t�z�v�s�g���������'�4�@�M�Y�_�a�^�M�@�4�'������������������'�(�0�(�����������������ּ��������������ʼ��T�Q�G�;�2�.�)�'�.�;�=�G�T�W�`�f�d�`�T�T����������������������!�%�#��������������������������������������������������������������������������������������������{�u�u�y��������������������������[�B�6�1�.�5�1�5�O�[�eāčĖĒčā�w�h�[�5�/�5�6�B�N�T�P�N�B�5�5�5�5�5�5�5�5�5�5�ù����������ùϹӹֹϹȹùùùùùùù��g�e�Z�Q�N�C�C�N�Z�g�g�o�s�t�s�m�g�g�g�g�����������!�+�&�!����������ɺĺźɺֺ޺���ֺɺɺɺɺɺɺɺɺɺɽ��������Ľнݽ����$�������ݽнĽ��4�/�4�?�A�M�N�O�M�A�4�4�4�4�4�4�4�4�4�4ùìàÜßïù�������'�4�5�)������ù�h�`�O�H�O�P�[�e�h�tāĆčđĚĘčā�t�h�I�>�=�<�=�A�I�V�Z�[�V�V�I�I�I�I�I�I�I�I�4�'�������'�4�@�Y�r�y��z�t�f�M�@�4�����������&�(�5�:�A�G�A�5�(���ŇŅ�{�n�b�W�U�R�U�b�n�w�{ŇŐňŇŇŇŇ����	����� �$�0�9�=�H�H�@�=�0�$���	����������!�����������ED�D�D�D�D�EEEE*E7ECEFEHEDE9E*EEE�!������!�'�.�1�:�G�K�V�T�S�G�:�.�!ĦěĚĕĘĚĢĦĳĿĿĿ����������ľĳĦ�����������������ĿѿӿҿӿѿϿĿ������� M ! 9 : | L 0 ; < J 6 D 5 Q $ e ( { $ 0 M = / # - 9 2 N w = K * ` ' i x 0 o N d  = I F G I 3 V H R R D J : M ) V g ) U  4 - J  �  �  �  G  k  J  )  �  �      �  �  E  �  {  r  p  -  �  �  0  a  �      �  �  m  &  �  >  �  �  �  c    �  �  �  j  �  h  �  M  J  [  �  �  $  �    �  o  �  �  l  �  v  T  �  ?  K  ;  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  S  J  @  3  &      �  �  �  �  �  �  q  Y  D  2  $  9  [  2  2  2  -  &      �  �  �  �  �  {  V  #  �  �  r  3   �  �       $  8  9  5  )      �  �  �  m  9    �  �  Z  $  h  c  _  [  V  R  N  G  ?  8  0  (           �  �  �  �  �  �  �  �  �  �  �  �  �          �  �  �  �  �  �  �  �  �  �  {  r  h  _  U  L  B  =  <  ;  ;  :  9  8  7  6  5      >  M  X  ]  b  u  �  �  �  �  k  F    �  �  r  1    3  H  _  |  �  �  �  �  �  �  e  @        �  �  U  �   �  �  �  �  �  �  �  �  �  �  �  �  �  l  L  "  �  �  b     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  m  c              �  �  �  �  �  �  w  V  3    �  �  X   �                �  �  �  �  �  �  �  �  �  n  I  $   �  �    4  @  D  D  =  -    �  �  �  �  l  /  �  �  :  }  �  >  O  a  j  o  r  k  W  8    �  �  �  �  ]  7  3  4  
  �  .  9  =  @  E  N  S  S  J  >  .    �  �  �  R  �  �  &   �  a  g  ]  K  7    �  �  �  �  �  O  	  �  Z  �  r  �  8  �  �  �  �  �  {  l  \  K  :  &    �  �  �    5  �  D  �   �  �  �  �  �  y  `  E  (    	  �  �  �  �  �  �  y  �  c    c  q  {  }  w  n  b  S  =  %    �  �  �  e  .  �    
   �  �  �  �  �  �  �  �  �  n  B    �  �  I  �  o  �  �  �  �  R  ~  �  �  �  �  �  �  �  v  W  0  �  �  x  "  �  i  8     �  �  �  �  �  �    x  p  i  `  U  I  4      �  �  �  �  �  �  �  �  �  �  �  l  U  =  !    �  �  �  �  g  5  �  �  �  �  �  �  �  
        �  �  �  �  o  !  �  b  �  �  /  �  �  �  �  �  �  �  �  �  r  A    �  c  !  �  ~  0  �   �  �  �  	  	,  	>  	N  	I  	2  	  �  �  2  �  ]  �  L    M  �  �  �  �  �  �  y  r  j  `  V  K  A  7  ,  !       �   �   �   �  �  y    [  �  �  �  �  m  G    �  @  �  �  �  w  
  '  h  �  �  z  p  d  W  C  /    �  �  �  �  l  D  *    �  �  �  �  �  �  �  �    9  I  O  L  9    �  �  f    �  l    ^  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  \  N  =  )    �  �  �  �  �  �  �  �  �  s  ]  I  ,  	    s  e  V  H  :  (    �  �  �  �  �  q  X  ?  (     �   �   �  W  8  e    z  j  Z  K  ;  -      �  �  �  �  ^    u  �  �  �  �  �  r  F    �  �  �  v  F    �  �  �  O    C  L  �  �  �  �  z  n  b  S  B  1        �   �   �   �   �   �   �   �  v  �  �  �  �  �  �  �  �  d  =    �  �  �  O  �  z  �                �  �  �  �  �  �  �  �  �  |  e  O  8  "  �  '  -    �  �  �  �  �  n  F    �  �  Y    �  L  �  ,  �  �  �  �  �  �  �  �  n  W  9    �  �  �  �  k  i  g  e  �  �  �  �  �  �  �  �  �  s  V  3    �  �  P  �  3  `  [  �  �  �  �  �  �  �  �  �  �  x  ]  B  %    �  �  u  L  -  �  �  �  �  �  �  j  Y  L  ?  .      �  �  �  �  �      	�  
  
"  
*  
)  
!  
  	�  	�  	�  	S  �  y  �  I  �  �  �  T  ,  n  h  O  C  1  )  #        �  �  �  �  %  �  w    �  *  U  N  F  ?  8  0  )  "          �  �  �  �  �  �  �  �  %  G  ^  s  �  �  �  �  �  z  X  2    �  �  L  �  �  y  @  �  ~  r  g  [  N  B  5  '      �  �  �  �  �  L     �   y  �  �  �  �  �  �  �  z  r  i  `  W  >    �  �  �  �  h  F  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  \  Q  K  E  A  ;  1  *  $        �  �  �  a  +  �  �  	  �  �  �  �  �  �  ~  k  X  D  /       �   �   �   �   �   �  �  	  	!  	  	  �  �  �  l  ?    �  �  |    �  �  �  �  �    
    �  �  �  �  �  c  '  �  �  j  *  �  �  D  �  �  R  <  '    �  �  �  �  ~  ]  <    �  �  �  U  �  �  >  �  z  �  �  �  �  �  g  8    �  �  ;  �  �    �  �  7  e  �  �  �  t  `  J  6  "    �  �  �  �  �  j  L  )  �  �  �  @   �  �  �  y  q  j  b  Z  R  J  B  >  =  <  ;  9  8  7  6  5  4  �  �  �  �  �  �  �  �  �  �  m  G    �  �  V    �  �  n  W  Y  �  3  I  Q  M  .    �  �  �  �  q  L  '     �  �  �     -  o  �  �  v  P    �  �    �  �  9  "  	�  �    [    �  �  �  �  �  �  �  �  �    c  8    �  �  u  H    �  �  �  �  |  p  e  U  A  *    �  �  �  p  =  
  �  �  i  S  y  E  0    �  �  �  ]  (  �  �  �  T  $  �  �  z  ;  �  �  h