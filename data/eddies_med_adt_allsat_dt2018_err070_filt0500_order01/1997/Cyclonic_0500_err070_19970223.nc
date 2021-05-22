CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�&�x���     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�b�   max       P�U�     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���
   max       <��
     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?(�\   max       @F�\(�     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v}G�z�     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @2�        max       @O�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�^          $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       <t�     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��[   max       B4��     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�2   max       B4��     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =	�9   max       C��T     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =%P�   max       C���     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          C     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          O     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          O     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�b�   max       P�U�     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�N;�5�Y   max       ?�����m     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���T   max       <�t�     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?#�
=p�   max       @F�\(�     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v|z�G�     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @O�           �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�R        max       @�^          $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         CD   max         CD     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���u��"   max       ?�����m     �  _�         	   
            	   	   1   C                           	         >                           5            	   )                  	   .                9   -      	            (                  	   2                     0         N�C�N\��NS͟O	O��MNq��N@��Nҁ�N��aP�U�P��RN�ϾO���O2�zN���N� N82O�4�O��N�
N�y�N�bkPn N�׌N{eNd8�N4�O�*AN�f6N��N(�nP>�NV�P8�9N�s8N6)FO�7�N5�wNe��O�ZO�?O���N�(WO8{�NX\�N2fNc[O�.O�8�O�P�Nk��N��;ON��M�b�ND�EP��O�hNB��O�<�N�2nN��N�6�PK�N))>P�aNLXNx/�O��O���O��NG��N��{M��<��
<e`B<e`B<t�<o<o;�`B;��
��o�t��49X�49X�49X�D���D���D���T���T���u��C����
��9X��9X��/��/��`B���o�o�o�o�+�C��C��C��\)�t��t��t��t��t������������#�
�#�
�,1�,1�,1�,1�8Q�8Q�D���H�9�H�9�H�9�P�`�P�`�P�`�T���T���ixսixսm�h�}󶽁%��O߽��������
���
26:?BLOZ[_hkjh[OB;62HHTUU\a\UH>DHHHHHHHH	)6796)						)5BDIFB:5*)#%&#����)=;7,)&!���>BBNNNSUXYNGBB>>>>>>����������������������������������������Y[hlt���th^[XYYYYYY����#Ob����{U<#����Vw��������������aVSUacnozz|zniaUTQSSSS�������51)����*/;HMMMLKLQHD;4/,''*���������������������������������������������������������������������������TT^amrrpppnmaTOKKOTTY[htx}uth[PTYYYYYYYY��������������������#-/<@EB<<;/.#!��0EOSMLQI<#

 ���HHU_aedbaUHGCAHHHHHH�������������������������������������������������������������������������������'/1<?HU^_^XUH<//''''��������������������v�������{wvvvvvvvvvv�������!$��������15BNQONB95+/11111111Sh�������������th[SSMOV[hjsnhf^[YSOMMMMM}�����������}}}}}}}}��������������������yz~����������zyyyyyy�����������������������������������������������
������������).6:AFB:5)���������

����������
#'/133.%# %)256BCCB;8761)&%%%%qt������tsqqqqqqqqqqENW[`b^[UNMFEEEEEEEE:BU]nz�������rrniZA:�����������������zz������

�����������������������������
!#&'#
��������Xent���������thc`[VX[[__ehjlnhh[[[[[[[[[������������ihnmt�����������tlfi��������������������/07<HIII<:0.,.//////hnkty����������tfcch��������������������ABGO[hkplgc[YOMLKGBA<BDN[]egng[NDB<<<<<<���&43-��������



����������#<In{����{bU=1,'X[bgmtttpg[SXXXXXXXX	 !"						��������������������X\]cgt���������t[WYX�����	����������������������������!#/3<=BEHJHD<0/&# !!NUacfba^URNNNNNNNNNN�����������������������ûлԻٻӻǻû������������������������������������������Һɺƺɺʺкպֺ�����ֺκɺɺɺɺɺ��g�b�a�Z�^�g�s���������������������s�m�g�� ��ѿ˿ʿ˿ѿ�����5�A�H�F�?�5���C�7�6�*�*�*�6�C�O�\�h�\�O�O�C�C�C�C�C�C�������ľʾ׾߾�׾ʾ��������������������������������������������ʾʾξʾʾ������e�]�_�e�n�r�u�~�������~�z�r�e�e�e�e�e�e�����������n�N�>�&�A�s���������	��
����	���������������������	�T�a�e�`�H�,��4�3�3�4�@�A�M�N�Z�a�\�Z�V�N�M�A�4�4�4�4àÓÇ�z�u�n�V�U�]�a�n�zÇÓÛàééåà���������������������	��	��������������0�.�$�"�� �$�0�=�G�E�>�=�0�0�0�0�0�0�0�(�!����!�(�4�4�A�B�L�M�O�M�A�;�4�(�(Ç�{�z�n�a�X�a�n�zÇÊËÇÇÇÇÇÇÇÇ�<�/��
����������#�/�<�G�U�V�V�Z�U�R�<��ƿƳƮƫƳ������������������������������������!�!�'�(�!�������������������������$�)�(�$������)�&�'�)�+�2�6�B�O�[�\�[�R�O�D�B�6�.�)�)�����{�x�������Ľݾ�4�@�9�8�4����Ľ�����������������������	�����������������ؼּּּ߼����������������ּּּ��a�U�Y�a�b�n�v�w�z�n�a�a�a�a�a�a�a�a�a�a�~�y�r�g�j�r�{�~���������~�~�~�~�~�~�~�~�C�6��������*�C�\�h�uƁƎƘƔƇ�u�h�C�������߾ܾ����	����	�	������������������������������������������������²«ª²¿��������¿²²²²²²²²²²�r�n�v�x����������!�3�:�9���ּ�����r��
�	�
����"�)�+�)������������m�^�W�h�z�����������������������������ù����������ùϹڹܹ����ֹܹϹùùù��<�;�6�<�@�H�U�\�Z�U�H�>�<�<�<�<�<�<�<�<�Q�R�J�I�N�_�l�x�~�����������������x�_�Q���������'�)�0�/�)���������B�;�6�0�2�6�@�B�O�O�O�I�O�R�O�D�B�B�B�B���Ϻɺ����ֺ��!�4�:�F�M�:�-�����ɺ������ź������ɻ�!�8�3� �����ֺ��[�O�D�;�6�6�D�O�h�tāčēĚěęĒ�t�h�[�	������������	����"�.�0�.���	EEEEEE*E7ECEPE\EiEkEiEbE\EPECE*EE�m�h�a�T�Q�L�T�a�b�m�z���������z�m�m�m�m�����������������������������������������#��#�'�/�<�H�O�H�B�<�/�#�#�#�#�#�#�#�#�;�/� ����������	��;�H�U�a�m�v�m�a�T�;�G�=�=�:�B�;�;�G�U�`�m�������������m�`�G�������������������Ŀѿݿ����ѿ�����ƚƚƚƧƧƳ��������ƳƧƚƚƚƚƚƚƚƚìêàÞÚàâìùþ����ýùìììììì�����������������żʼͼ޼����ּѼʼ��y�w�m�`�T�Q�T�`�m�y�y�z�y�y�y�y�y�y�y�y����������������������¦�t�[�X�e�g�t¦²��������������¦�Ŀ����Ŀſѿҿݿ�������������ݿѿĿĻ_�^�_�a�l�w�x�x���������x�l�_�_�_�_�_�_Ň�n�b�O�U�^�nŇŠŭ����������������ŭŇ�=�3�3�=�I�V�]�b�c�b�V�I�=�=�=�=�=�=�=�=����������������������������������������ÓÉÇ�ÂÇÓÛàäááàßÓÓÓÓÓÓ�!��!� �.�:�S�������ʽսĽ������l�S�.�!���ܻܻܻ�����������������û������������м��)�4�2�-�����л����������� ����������������������(�4�?�4�0�(���������������������
��#�0�<�@�H�<�0�&�#��
��������ĳĭģĠıļ���������������������޻ݻ�����'�@�M�a�d�Z�K�5������������������ûлٻллû���������������FF	F	FFFF$F1F<F=F?F=F<F1F0F$FFFF�����������ùĹù����������������������� O D v P = h . / [ w B c J f D * | R [ 9 0 m A % j E H d > V ) W O H ; O 9 u L E F N v Q z D ] @ 0 1 K H ' n W 8 P A Y + C M J  @ < m ` 9 X P j A    &  d  �  W    �  \  �  �  p  �  �  �  �  �  �  v  <  _  �  �      �  �  �  Q  x  ,  Z  @  �  r  T  �  n    z  �    X  �  >  �  �  V  �  .  �  @  �  �  �  D  �  �  7  j  [  �  $  �  �  m  �  R  �  s  `  s  _  �  "<t�<t�;��
%   ��o;D��;D���D���T���ixս��-����+����9X��C�����t��o���ͼ�`B����1�+�@���w�t��e`B�49X�\)��㽰 Ž�w��%�P�`�49X����,1�0 Žu�}󶽃o�@�����''8Q콑hs�ȴ9�� ŽD���P�`��7L�@��e`B��9X����P�`��t��aG��q���u���`�y�#���-�}󶽅���Ƨ�����w�ě����B��B��B�^B0$B��BQ�BrbB4��B��B%y�BZYBk�B��A��[B�@B�7B)BR�A���B��B
qB QB$��B�}B�dB"GB!�kBnwB�BiB�aB-F�B�B~�Bb�B
��B!#B \>B}-B�BB"�B
JBIBz�B��B�BdB�B�jB�YBN�B�BmYB��Bw�B
�B��B&rB
nhBz�B�BCBB6B$(B'#UB	+�BoB�B
�7BCrB"�BGB�|B@<B�B�;B=�B\VBB�B`�B4��B�PB&��B��B> B7�A�2BЌB9�BAIB��A�oFB�B��B�B%5;B��B��B"qB!��B?�BE�B�B�cB-CB��B
�BF�B
�8B ��B >�BMB?\B#]MBB?;BNpBKtBűBJ�B:�B��B�B3�B>BD�B�aB�9B
�/B��B&.B
�1B��B5AB<�B;jB#��B'TxB	6�BG-B�
B
�B�B�B?�B�@��A�z\@: A�1A��RB �AQ�
AMCM?���A�B�A�E#A<%A�gvA��B
G�A8�mA��OA��B�@d��B��A�J�A&%�A�5�A�7A�.@��B�AXW�A�FA��A-A�E�A���>~R�A�{C@�>>A��A�F�@T�y@L�A۶4AY��C���A�a�A *�A��A�-�Aj(sAw��B.�A��@�lAj�A���A�:A}}@�)�A�d�BZkA���A�d�ArV@���@�)�A��A4�A�T-A��@�8~@�M�C��T=	�9@��A�h�@<�A�}XA��XB2�AQ�1AMK�?�T�A��A�Q$A;#�A��A�h�B
@yA:�LA�x8A�;�B�@c�sB	7�A׀�A$��A��A$A��?��pB \!AX�A�u�A�z�AvAԆ�A��>QX�A�~�@�:�Aԁ�A�p�@Xa
@S��Aڠ�AX�)C��FA���A ^/A�q�A���Aj��Ax�EBFFĀ�@�e�Ai�GA��fA�j�A}'$@���A��7BD�A���A�tlA��@���@�#�A�q}A5�lA��A�f�@�M@��C���=%P�         
   
            	   
   1   C   	                        	   	   	   ?                           6            
   )                  
   .                9   -      
         	   (               	   	   3                     0                        #               O   ;                                    5               #            7      -                  %   )   #                  '   #                     )         '            1      +            %   '                                       O                                                      #            3      -                  #   )                     !                        '         '            +      +            %   '         N��N\��NS͟N��zN�:5Nq��N@��N���N��aP�U�O�AN�ϾO���N���N���N� N82O�4�O��N�
N�y�N�bkO��N�׌N{eNd8�N4�O�*AN�f6N��N(�nP9N�NV�P/ʗN9�N6)FO3��N5�wN6ocO���O�?OwEN��rN�@�NX\�N2fNc[O͠�OazGO��.Nk��N��ON��M�b�ND�EP�]N�өNB��O�<�N�2nN��NK�VP&}�N))>P�aNLXNx/�N�i�O���O��N��Nn��M��  �  /  t  <    �  T    �  �  -  �  �  �  G    �  �  5  g  u    �    	�  ,  i  �  �  ?  �    �  �  �  �  �  �  �  A  .  X  q  
�  6  �  �  �  �  o  =  '  w  �  �  +  �  R  (  �      �  �  A  �  g  e  	  �      k<�t�<e`B<e`B;�`B�ě�<o;�`B;D����o�t��8Q�49X�49X��o�D���D���T���T���u��C����
��9X�P�`��/��/��`B���o�o�o�o�C��C��\)��w�\)�49X�t���P��w�t��0 Ž�w�@������#�
�,1�u�<j�,1�0 Ž8Q�8Q�D���P�`�aG��H�9�P�`�P�`�P�`�Y��m�h�ixսixսm�h�}󶽅���O߽�����P���T���
BBCOV[\hjhh[ONB@6=BBHHTUU\a\UH>DHHHHHHHH	)6796)						&)5ABFCB=553+)%#&&&&��������>BBNNNSUXYNGBB>>>>>>����������������������������������������Y[hlt���th^[XYYYYYY����#Ob����{U<#������������������������SUacnozz|zniaUTQSSSS�������51)����-/6;HIJIHHIH;8/.**--���������������������������������������������������������������������������TT^amrrpppnmaTOKKOTTY[htx}uth[PTYYYYYYYY��������������������#-/<@EB<<;/.#!� 
#0<?@=:0#
�����HHU_aedbaUHGCAHHHHHH�������������������������������������������������������������������������������'/1<?HU^_^XUH<//''''��������������������v�������{wvvvvvvvvvv������� $��������15BNQONB95+/11111111Ti��������������t[VT[[^hokh][ZQP[[[[[[[[}�����������}}}}}}}}��������������������yz~����������zyyyyyy�����������������������������������������������
�����������)*27<;5)����������

���������	
 #*./0//)#
		%)256BCCB;8761)&%%%%qt������tsqqqqqqqqqqENW[`b^[UNMFEEEEEEEEFU_nz���������zaG?=F��������������������������

�����������������������������

#%&#
��������Xent���������thc`[VX[[__ehjlnhh[[[[[[[[[������������gjimtx�����������tig��������������������/07<HIII<:0.,.//////hnkty����������tfcch��������������������ABGO[hkplgc[YOMLKGBA=BFN[[c[YNIB========���"'+,)��������



����������#<In{����{bU=1,'X[bgmtttpg[SXXXXXXXX	 !"						��������������������X\]cgt���������t[WYX�����	����������������������������"#/2<<AC<2/'#!""""""NUacfba^URNNNNNNNNNN�����������������ûлл׻ллĻû����������������������������������������������Һɺƺɺʺкպֺ�����ֺκɺɺɺɺɺ��g�f�_�g�g�s�������������������s�g�g�g�g�� ��������������'�!������C�7�6�*�*�*�6�C�O�\�h�\�O�O�C�C�C�C�C�C�������ľʾ׾߾�׾ʾ����������������������������������ľʾ˾ʾž����������������e�]�_�e�n�r�u�~�������~�z�r�e�e�e�e�e�e�����������n�N�>�&�A�s���������	��
����	�� ��
��"�/�;�H�T�V�X�S�H�;�/�"��4�3�3�4�@�A�M�N�Z�a�\�Z�V�N�M�A�4�4�4�4àÓÇ�z�u�n�V�U�]�a�n�zÇÓÛàééåà������������������������ ���������������0�.�$�"�� �$�0�=�G�E�>�=�0�0�0�0�0�0�0�(�!����!�(�4�4�A�B�L�M�O�M�A�;�4�(�(Ç�{�z�n�a�X�a�n�zÇÊËÇÇÇÇÇÇÇÇ�<�/��
����������#�/�<�G�U�V�V�Z�U�R�<��ƿƳƮƫƳ������������������������������������!�!�'�(�!�������������������������$�)�(�$������)�&�'�)�+�2�6�B�O�[�\�[�R�O�D�B�6�.�)�)�������������������Ľн������޽нĽ�����������������������	�����������������ؼּּּ߼����������������ּּּ��a�U�Y�a�b�n�v�w�z�n�a�a�a�a�a�a�a�a�a�a�~�y�r�g�j�r�{�~���������~�~�~�~�~�~�~�~�C�6��������*�C�\�h�uƁƎƘƔƇ�u�h�C�������߾ܾ����	����	�	������������������������������������������������²«ª²¿��������¿²²²²²²²²²²����x�y����������!�2�:�9����ּ�������
�	�
����"�)�+�)������������m�^�Y�h�z�������������������������������������ùϹչܹ޹ܹϹù����������������<�;�6�<�@�H�U�\�Z�U�H�>�<�<�<�<�<�<�<�<�h�_�Z�S�U�_�_�l�x�������������������x�h���������'�)�0�/�)���������6�1�3�6�B�B�E�G�O�Q�O�B�6�6�6�6�6�6�6�6���Ժɺź��Ǻֺ����%�-�?�:�5�-����ɺ������ź������ɻ�!�8�3� �����ֺ��O�K�B�:�?�B�K�O�h�tāčēēĎā�t�h�[�O�	������������	�����
�	�	�	�	E*E%EEEEE*E7EAECEPEPE\EdE]EPECE>E7E*�m�h�a�T�Q�L�T�a�b�m�z���������z�m�m�m�m�����������������������������������������#��#�'�/�<�H�O�H�B�<�/�#�#�#�#�#�#�#�#�/�"������������	��;�H�R�U�]�`�T�H�;�/�T�I�H�E�H�T�V�`�m�y�~�����������y�m�`�T�������������������Ŀѿ�����߿ѿ�����ƚƚƚƧƧƳ��������ƳƧƚƚƚƚƚƚƚƚìëààÛàæìùýÿýùôìììììì�����������������żʼͼ޼����ּѼʼ��y�w�m�`�T�Q�T�`�m�y�y�z�y�y�y�y�y�y�y�y������������������������¦�e�g�i�v¦²�������������˿ѿпĿÿĿɿѿ׿ݿ�����������ݿѿѿѻ_�^�_�a�l�w�x�x���������x�l�_�_�_�_�_�_Ň�n�b�O�U�^�nŇŠŭ����������������ŭŇ�=�3�3�=�I�V�]�b�c�b�V�I�=�=�=�=�=�=�=�=����������������������������������������ÓËÇÀÅÇÓßàáàÝÓÓÓÓÓÓÓÓ�!� �$�$�'�.�G�S�y�������������z�f�S�.�!���ܻܻܻ�����������������û������������м��)�4�2�-�����л����������� ����������������������(�4�?�4�0�(�����������������������
��#�+�0�<�=�<�0�#� ���������ĳĭģĠıļ���������������������޻ݻ�����'�@�M�a�d�Z�K�5������������������ûлջл˻û���������������FF
F	FFFF$F1F:F1F0F$FFFFFFFF�����������ùĹù����������������������� ] D v > , h .  [ w & c J ^ D * | R [ 9 0 m  % j E H d > V ) P O I $ O 9 u N : F J E A z D ] ? % 1 K C ' n W 3 = A Y + C : D  @ < m P 9 X I ^ A    �  d  �  �  �  �  \  �  �  p  2  �  �  2  �  �  v  <  _  �  �    e  �  �  �  Q  x  ,  Z  @  �  r  4  N  n  �  z  j  �  X  +  �    �  V  �  �  �    �  �  �  D  �  f  �  j  [  �  $  h    m  �  R  �  ;  `  s  A  �  "  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  CD  �  �  �  �  �  �  �  {  k  W  <  )    V  �  v  c  Q  @  /  /  $          �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  q  o  u  {  k  U  ?  )      �  �  �  �  �  �  �  �  �    (  3  7  ;  :  7  3  .  '      �  �  �  �  �  p  S  5  �  �  �  �      �  �  �    
      �  �  �  �  r  F  �  �  �  �  z  n  _  Q  B  0      �  �  �  �  [  E  2      T  O  J  F  A  <  6  0  +  %                      �  �  �            �  �  �  �  �  �  �  �  �  b  =    �  �  �  �  �  �  �  �  �  l  P  3    �  �  �  G  �  �  c  �  �  �  �  �  ^  $  �  �  2  �  p    �  �  �  I  �  �  4  �    8  c  �  �  	      $  )    �  �  k    �  �  �  �  �  �  �  �  �  �  �  �  �  }  i  Z  I  *    �  �  x    �  �  �  �  z  R  &  �  �  �  �  k  I  )  	  �  �  �  E  �  B  �  �  �  �  �  �  �  �  �  j  <    �  �  b  &  �  �  \    G  B  <  -      �  �  �  �  �  o  L  &  �  �  �  0  �  �         �  �  �  �  �  �  �  �  �  �  �  }  i  ,   �   �   V  �  y  l  `  M  9  %  �  �  /  �  �  �  S    �  �  t  :     �  �  t  N  &  �  �  �  g  ,  �  �  T     �  Y    �  U  s  5       �  �  �  �  o  C    �  �    B    �  i    �  �  g  Y  K  A  8  0  '        �  �  �  �  �  o  "  �  �  C  u  ]  E  -    �  �  �  �  �  �  u  ]  G  1      �  �  �      �  �  �  �  �  �  �  h  E  "  �  �  �  �  x  b  d  g  �  �  �  �  �  �  _  �  �  �  �  �  n     �  ?  r  �    �       �  �  �  �  �  �  �  z  i  X  G  6  %          %  	�  	�  	~  	u  	�  	�  	l  	2  �  �  z  5  �  �  g  �  �  o  �  a  ,      �  �  �  �  �  q  ]  T  F  5  #    �  �  �  �  �  i  c  ^  X  R  I  @  7  /  )  "            :  b  �  �  �  �  �  �  �  �  z  [  6    �  �  �  f  V  ;  �  �    +  �  �  �  �  �  �  �  �  �  �  x  [  <    �  �  �  b  &  �  ?  D  H  M  R  V  [  X  S  M  G  A  ;  6  0  *  $        �  �  �  �  �  �  �  �  �  �  �  |  m  ^  N  =  ,       �  �  �  �  �  l  ;  
  �  �  r  H    �  �  K  �    �     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  b  Z  R  D  +    �  �  q  -  �    �  d  |  �  �  �  �  �  �  �  �  �  �  �  p  U  6    �  �  �  �  �  �  �  �  �  �  �  �  }  o  _  O  =  *      �  �  �  �  `  �  �  �  �  �  �  �  �  �  v  Q    �  �     �  ;  �  �  �  w  ^  E  #  �  �  �  x  E    �  �  �  �  Y  +   �   �   �  �  �  �  �  �  �  �  �  �  �  �  l  N  0    �  �  �  d  9  3  <  ?  >  *    �  �  �  b  +  �  �  �  t  J    �  �  ^  .    �  �  �    w  �  �  �  �  r  M    �  �  F  &  $  *  ;  I  S  X  W  O  >  )  
  �  �  x  ?  �  �  m  3  �  !  /  =  U  n  ]  G  .    �  �  �  �  �  �  �  p  U  7    �  j  	�  
  
�  
�  
�  
�  
�  
�  
M  
  	�  	�  	K  �  �  ^  	  �    �  6  1  -  (  #            �  �  �  �    %  ?  Y  s  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  a  �  �  �  �  �  z  l  ^  P  D  8  ,  1  A  R  b  j  p  u  {  �  �  �  �  �  �  �  �  �  u  L  %  �  �  �  Z  !  �  v  �    �  �  7  o  �  �  �  �  w  D    �  _  �  \  �  z  V  X  ^  k  m  _  F  "  �  �  �  _  "  �  �  &  �  -  �    y  �  =  0  "      �  �  �  �  �  �  �  ~  f  E  #    �  �  �    "  '  %  #          �  �  �  ~  C  '  
  �  �  �  �  w  ]  B  $    �  �  �  �  i  R  J  L  S  ]  t  �  �  �  $  �  q  ]  J  6  "     �   �   �   �   �   �   �   �   �   �   x   m   b  �  �  �  �  �  �  n  U  C  3  (  #      �  �  �  �  �  y    +  #    �  �  �  �  c  @  ,    �  �  G  �  �    �  �  �  �  �  �  �  �  �  �  v  M    �  �  Y    �  !  �    ]  R  K  E  >  8  1  +  $         �   �   �   �   �   �   �   �   w  (    �  �  �  �  �  �  �  �  �  �  n  N  #  �  �  k  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  Z  @  %      v  l  c  Z  O  ?  /    	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  _  D  )    �  �  �  E    �    �  �  �  t  R  1    �  �  �  D  �  �  t  ;  �  �  /    �  �  �  �  �  �  |  t  l  c  X  H  8  )    
  �  �  �  �  A  (    �  �  �  �  �  n  O  2      �  �  �  s  /     �  �  �  �  �  �  �  x  k  ^  Q  ;    �  �  �  �  M     �   �  g  ^  U  L  C  9  0  (  !          
  -  O  q  �  �  �  N  ]  e  V  K  7  $  !    $  .  h  R  8    �  �  �  W  !  	  �  �  �  }  T  :  4  G  Q  T  L  *  �  �  f    �  "  y  �  �  �  r  G    �  �  b    �  ]  +  ;  �  `  �  *  ^  �                    �  �  �  �  �  �  �  �  �  �  �  �    �  �  �  �    d  �  �  H    �  {  2  �  �  K  �  �  k  b  Z  Q  H  @  7  /  &          �  �  �  �  �  �  �