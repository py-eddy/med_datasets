CDF       
      obs    G   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?����E�       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N
s\   max       P��>       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��Q�   max       <��
       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��
   max       @FǮz�H       !    effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �׮z�H    max       @vnz�G�       ,   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P�           �  70   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���           7�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �"��   max       �D��       8�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B4�@       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��f   max       B4�x       ;   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       <SF�   max       C��n       <0   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�pE   max       C��A       =L   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          b       >h   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =       ?�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          =       @�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       NU�   max       P��>       A�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��!�.H�   max       ?�\(��       B�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       �t�       C�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�����   max       @FǮz�H       E   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��
=p��    max       @vk��Q�       P(   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P�           �  [@   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���           [�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @@   max         @@       \�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���҈�   max       ?�\(��     0  ^   \                  S   !                  $   5   %            )                  ;         /      !      C   	               <   
            $                              L      
      	      a      !   
                        #P�N_�N�IN�|BO��O�Pbe�O��O��N�9VO*O�	&NSn�O�bP��>O-�N��BN/�N�<O�p[N�AMN�I�N!-NN�l�OLM:P,��OYUN��CO�#)N��O�,�O��WO�p�N��O��N:|vOW)�Nh,�O��{N�- O{�9O��N���O��/O(�N
s\N*�PO6�O��N��vN*�O(��O.Pw�N�h�N��O}�N�,�N��8O�O�whO�q�N�L�O�vMO�JN��iN6N4b"N���N�[GOs�<��
�t��t��t��49X�D���u�u��o��o��C���C���t����㼛�㼼j���ͼ�������������/��/��`B�������o�o�o�o�+�\)�\)�\)�\)�\)��P��P�����'',1�0 Ž49X�8Q�D���D���H�9�H�9�H�9�H�9�L�ͽP�`�T���aG��e`B�q���q���}�}󶽅������t����㽛�㽛�㽡��������Q콸Q�%6O[c|{ll[OB6)"�����
�������������st�������yttssssssss������������������������������������������������
�������mz�������������tihmeky�������������tnge-5BNg�������tiZA;:0-��������������������)*-..,)�'*5MV[ghjjg[NB;5/-('rt�����tnorrrrrrrrrrNgt������������g[QLN�#0<{�����raM<0
������������������������������������������������������	
##%#
								BO[h��������th[OF?=B�����������������������������������������
##'#
������������������������������&)6BJOQPFB>6)(0[hty|����h[OKMG=*("-/;DHT`TPHG=7/"! !"����������������������������������������;<HUXZUHC<;;;;;;;;;;�������
���������������������������
#/JQVLF<</#HHQU\aknrrqnga_UHFHH�����������������wx�������������������������������������������������������������!'/Hatz|uvnaU<#lnrqzzz�����}zunmlll����������������������������������������X[`hltzzthg[TVXXXXXX��&)6BLJE6)�������������������������������������������!#/<C@</#"!!!!!!!!!!qz���������������}zqaqz����������zmc_]^a��������������������������������������������������������������������������������! ����������� �����������������������������fgnt�����������tngff/56BN[TQNB54////////��������������������<BMPKH</
����
#/<V\gt�����������tg[VV���)7;<86)��������>GV]fn{}}�}wnbUI<2>gtw��������������tpgCHUVafnonnda\UHCCCCCzz������zrzzzzzzzzzz,0<@@><0%',,,,,,,,,,DIUVWUURNJICA@A@DDDD�����������������������������������������x�l�R�F�@�D�J�S�x�����ûջջлŻ������x�s�s�g�d�c�g�s�|�x�s�s�s�s�s�s�s�s�s�s�s�H�D�F�H�U�Y�a�b�e�a�V�U�H�H�H�H�H�H�H�H�U�S�H�D�<�8�<�A�H�N�U�a�f�e�e�b�a�V�U�U���������������������������������������������y�w�m�e�`�a�f�m�y����������������������ƳƧƜƎƙ�������$�/�7�3�0�$���������������¾�������������
���#�$���
���z�s�����x�~���������������������������z��������������� �"������������������������������!�!�!�������������������������������������������ƧƠƟƧƳ��������ƳƧƧƧƧƧƧƧƧƧƧ�T�F�C�;�6�4�I�Q�V�b�oǈǎǒǑ��z�o�b�T���|�b�S�J�3�5�Z�s�����������������������ʼɼ������������ʼּ�����������ּʺe�^�Y�O�K�@�:�/�/�3�@�L�Y�e�e�]�_�e�g�e������������������������������ùøìêììùþ��������ùùùùùùùù�\�M�E�A�=�G�T�`�y�������������������m�\�b�Y�Z�b�n�x�{�~ŇŔŗŔŇ�{�{�n�b�b�b�b�f�Y�S�E�;�4�.�4�9�@�G�M�Y�]�f�k�l�l�f�fŹŵŷŹŹ����������ŹŹŹŹŹŹŹŹŹŹ�Ŀ������������ĿͿѿӿݿ޿ݿݿѿĿĿĿ��m�g�U�H�D�;�6�;�A�H�T�m�t�z���������z�m�����|�}�����������)�-�6�6�/�	���������	�����������������	���"�/�:�6�/�"��	�m�g�l�m�q�w�y���������������������y�m�m���پ̾ƾǾξ׾��	��!�"�%�"��	�����g�f�[�c�g�s�x�|�s�h�g�g�g�g�g�g�g�g�g�g������������������$�=�I�X�\�V�I�$�����	�� ���/�T�m�z�~���~�v�f�S�4�/�"��	E\EPE@E@EGE\EiE�E�E�E�E�E�E�E�E�E�E�EuE\�"�!���	�������	���"�+�.�6�0�.�"�"�ʾ��������������ʾ׾����������ʾf�c�b�f�s�����������s�f�f�f�f�f�f�f�f�Y�Q�L�H�F�F�L�Q�Y�e�r�����������~�r�e�Y���������������ɺֺ׺ںֺ̺ɺ������������n�M�A�8�0�.�4�A�M�s������������������n���������������������ʾ׾���ܾ׾ʾ������������������Ŀѿݿ�����
����ݿѿ��y�v�m�`�\�T�V�`�m�y�����������������y�y��������'�-�3�3�?�3�'��������������x�t����������ͼּ���ټ����������������������������ùѹ׹ӹϹɹù����ּּռռּ�����ּּּּּּּּּ��H�E�A�D�H�U�X�`�V�U�H�H�H�H�H�H�H�H�H�H�ѿɿƿĿ����Ŀǿѿݿ�����������ݿۿ��Y�A�;�5�7�@�A�N�Q�g�s�������������s�g�Y�A�:�5�4�5�6�A�N�W�Z�^�]�Z�N�A�A�A�A�A�A��ܼۼ����������������������㾥���������������������Ǿʾ;ξ˾ʾ�����ĳıĬĭĵ�������������
�����������Ŀĳìæãàìù��������6�<�9�2�)����ùì������
����$�*�*�0�6�7�6�*�����������������)�,�4�)�������������������������
��#�%�#�!����
���*������*�6�;�@�8�6�*�*�*�*�*�*�*�*�U�T�I�<�:�9�<�I�U�b�d�n�w�n�b�X�U�U�U�UD{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDqD{�w�y¦¿������������¾·¦��н������������Ľݾ��"�)�(�������:�2�:�G�H�S�`�i�l�y��y�l�`�S�G�:�:�:�:�����������}�z�������ûлܻ�����޻ѻл��<�5�/�%���
���
��#�/�<�A�H�P�P�I�H�<�����������������������������������������z�z�n�n�z�����������z�z�z�z�z�z�z�z�z�z�������������������������������������������ĽĽнݽ����ݽнĽ����������ɺɺ��������ɺֺ�����������ֺɺɻ����!�-�5�:�S�l�x��������l�S�:�-� . m U S � 9 % 1 X / > F 2 T S * R � O  ^ g v J O h j L  B Y x ] t - 2  0 ? q : - B E ( n ? i I 6 � 4 [ P : v ( 1 Z T > E _ I w u ] > w Q _    �  g  D    �  F  �  N  �    ,  �  X  l  �  u    �  4  �  �  h  �  �  �  �  �  	  9  D  �  �  r  �  ?  ]  �  v  `         �  �  `  d  X  �  K  �  �  p  >  �    �  >  �  �    �  [  �  y  �  �  G  =  �  �  P��hs�D���D����h��t��'ȴ9�D���t���P���ͽo��1�aG���hs�q���t�����h��7L���'+�\)�Y���9X�49X��㽡����P��+�y�#���`�0 ŽY��']/�@��ȴ9�@���o�]/�]/������7L�D���Y���\)��+�y�#�T����t���7L�J�q����o��\)��C���7L�"�彮{�ȴ9���P��9X�ě����
���
��{��{���`���B�=B��B�B �\B�B�B�wBt�B	BB1kB�B
vB
B
�B&i�B�B��B�oBD;B�^B�nB!�B	�B^�BoqBypA��B�BX�BP�B�B+B�B
?BͧB ��B!�eB!EdBt�BP.B*��B+-�B�iB�+B��B*SnBMWB��A���B�,B,�B4�@B��B�B��B��B
D/BөBr?B��B
8�B��B�B'�=B
�(B��B�B%�B&�BLOB��B��B�B�aB �B��BѬB��BCBЏB;B@B�MB
7�B
��B&TrB@=B�jB�B}B�~B��B!n BK�B!�B�hB<1A��fB��BC�BG;B@�BE�BA�B�BV�B �OB!�B!?�B�9BH7B*�{B+:iB�4B��B��B*��B@~B��B 2B�kB-:/B4�xB�B�AB@B��B
>"B�B?�BA�B
�B��B��B'�aB
�OB�+B��B%�vB&��B>�B�@�TA�nPA�lOA�'4A���An܊B�A�^CA�T�@���A���A�lB<aB�jA���A �l?�iLB��A͒�Am� A��C@��A�̤Ay��A�S�A��yA�liAng�AWJ�A��~B	�MA�܌C��nA\ֈASԞAB��?�K@0WvA>2aAO:�A|��Al��?��@�'�<SF�AԏA��+A~w�A�@�A��A=�AL��A�sAў�A�[&A��A�&�A�Z�A�JC��A��A+�A�@��JA���A��7A��UA1j�A'�g@@M�@�&Z@�kA�h�Aş�AĚA���An��B��A�zjA�I�@��wA��aA�Y�B�B�A���A�?���B��A�e|An��A��@���A�s�Ay OA�RcA�wA�NAmaAV��A���B
MmA���C��A_MAT�AB��?�C�@0�A=APlFA}
�Al)�?�pE@���C��AA�A�= A}�A�GA�1AAL�	A�A� �A��A��A��A���A���C��A�jA-�A�2@�=(A�jNA�z�A��UA2,�A)�x@=��@�/F   \                  T   "         	         %   5   %            *                  ;         0      "      D   	               =   
            %                              M      
      
   	   b      "   
                        #   +                  -      #               %   =               !                  3               %   %   )                  %               #                              %                  "      '      !                                          #                     !   =                                 !                  %   !                                                               %                  "      #                           OW�RN_�N�INm8O��OWtPN.:�Oa�N�9VO*O�	&NSn�O���P��>O$+N��N/�N�<OP��N[�N�I�N!-NN�l�OLM:O��VN��N��CO4��N��On��O��WO�_N��O��NU�OW)�N)�(O@�VN�- OgB�O��N���OG��O��N
s\N*�PO6�O��N��vN*�O(��Ol��O��N�h�N��O}�N�,�N��8O�O��NO�*�NiIO�qBO�JN��iN6N4b"N���N�[GO`G�  	�  �  ?  �  �  q  z  '  �  s  �  y    3    �  9  @  5  c  �  T  �  -  �    �  �  I  �  2  4  
�  �  �  �  �  �  �      �  �    �  �  b  �  	  �  �  �  ,  m    .  ;     �  �  2  r  �  @  (  n  h  �    �  ���ͼt��t��e`B�49X�e`B�t��0 ż�j��o��C���C���t���j����ě���/���������#�
��`B��/��`B�����Y��C��o�49X�o��P�\)�<j�\)�\)�t���P��w�y�#���,1�',1�]/�<j�8Q�D���D���H�9�H�9�H�9�H�9�P�`�ixսT���aG��e`B�q���q���}󶽁%��hs��+������㽛�㽛�㽡��������Q콼j36@BO[dihe^[VOB;7333�����
�������������st�������yttssssssss����������������������������������������������
	���������u��������������wssu��������������������MNY[gtx�����tg[NNEMM��������������������)*-..,)�'*5MV[ghjjg[NB;5/-('rt�����tnorrrrrrrrrrR[t������������g[SOR�#0<{�����raM<0
�����������������������������������������������������	
##%#
								JO[ht�����uth[VOKHHJ�����������������������������������������
##'#
������������������������������&)6BJOQPFB>6)@M[horvxx{xthc[TOG?@!"%/1;HOLHD;;5/""!!!����������������������������������������;<HUXZUHC<;;;;;;;;;;�����
����������������������������#4AJOPG@:/#HHQU\aknrrqnga_UHFHH�����������������wx�������������������������������������������������������������-/9HTVamqnnfaUH</.*-lnrqzzz�����}zunmlll����������������������������������������X[`hltzzthg[TVXXXXXX�)6>BDA96-) ������������������������������������������!#/<C@</#"!!!!!!!!!!qz���������������}zqaqz����������zmc_]^a������������������������������������������������������������������������������������
���������� �����������������������������fgnt�����������tngff/56BN[TQNB54////////��������������������<BMPKH</
����
#/<W]gt�����������tg[WW��)36874)���������������IXin{||�~|vnbUI==AIgtw��������������tpgCHUVafnonnda\UHCCCCCzz������zrzzzzzzzzzz,0<@@><0%',,,,,,,,,,DIUVWUURNJICA@A@DDDD�����������������������������������������l�f�_�^�`�i�x�����������������������x�l�s�s�g�d�c�g�s�|�x�s�s�s�s�s�s�s�s�s�s�s�H�D�F�H�U�Y�a�b�e�a�V�U�H�H�H�H�H�H�H�H�H�G�<�;�<�E�H�T�U�V�`�a�a�a�]�U�H�H�H�H�����������������������������������������y�y�m�f�a�b�h�m�y���������������������y����Ƶƴƿ�������$�(�-�,�'����������������������������
�
�
�
��������������������������������������������������������������������� �"������������������������������!�!�!�������������������������������������������ƧƠƟƧƳ��������ƳƧƧƧƧƧƧƧƧƧƧ�V�P�E�?�;�=�I�Q�^�b�o�{ǋǏǏ�~�w�o�b�V���|�b�S�J�3�5�Z�s�����������������������������������ʼּ���������ּʼ����@�>�3�3�1�3�@�L�Y�c�Y�Y�L�C�@�@�@�@�@�@������������������������������ùøìêììùþ��������ùùùùùùùù�m�f�\�Z�^�g�m�y���������������������y�m�b�\�[�b�n�{ŇőŇ�{�x�n�b�b�b�b�b�b�b�b�f�Y�S�E�;�4�.�4�9�@�G�M�Y�]�f�k�l�l�f�fŹŵŷŹŹ����������ŹŹŹŹŹŹŹŹŹŹ�Ŀ������������ĿͿѿӿݿ޿ݿݿѿĿĿĿ��m�g�U�H�D�;�6�;�A�H�T�m�t�z���������z�m���������������������	���	��������������	�����������	���"�"�/�8�1�/�"���m�g�l�m�q�w�y���������������������y�m�m���վξξ׾۾����	�	����	��������g�f�[�c�g�s�x�|�s�h�g�g�g�g�g�g�g�g�g�g������������$�0�=�I�V�Z�V�S�I�=�$���	�� ���/�T�m�z�~���~�v�f�S�4�/�"��	EPEEEDEPE\EiEuE�E�E�E�E�E�E�E�E�E�EuE\EP�"�!���	�������	���"�+�.�6�0�.�"�"�ʾ��������������ʾ׾����������ʾf�e�c�f�s������s�f�f�f�f�f�f�f�f�f�f�Y�Q�L�H�F�F�L�Q�Y�e�r�����������~�r�e�Y�����������ɺкֺ׺ֺɺȺ����������������M�I�A�8�4�4�7�A�M�Z�[�f�m�r�t�n�h�f�Z�M���������������������ʾ׾���ܾ׾ʾ����ÿ��������Ŀѿݿ������	�����ݿѿÿy�v�m�`�\�T�V�`�m�y�����������������y�y��������'�-�3�3�?�3�'���������������������������������¼ȼ̼ü����������������������������ùιϹչҹϹù����ּּռռּ�����ּּּּּּּּּ��H�E�A�D�H�U�X�`�V�U�H�H�H�H�H�H�H�H�H�H�ѿɿƿĿ����Ŀǿѿݿ�����������ݿۿ��Y�A�;�5�7�@�A�N�Q�g�s�������������s�g�Y�A�:�5�4�5�6�A�N�W�Z�^�]�Z�N�A�A�A�A�A�A��ܼۼ����������������������㾥���������������������Ǿʾ;ξ˾ʾ�����ĳĭĮĳĶ�������������	�����������Ŀĳùñêèéìø��������6�7�5�.�)����ù������
����$�*�*�0�6�7�6�*�����������������)�,�4�)�������������������������
��#�%�#�!����
���*������*�6�;�@�8�6�*�*�*�*�*�*�*�*�U�T�I�<�:�9�<�I�U�b�d�n�w�n�b�X�U�U�U�UD{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DoDqD{�x�z¦­¿����������½´¦����н������������Ľݽ����$�#������G�:�G�M�S�`�f�l�x�l�`�S�G�G�G�G�G�G�G�G���������}���������ûлܻ����ݻлû����<�5�/�%���
���
��#�/�<�A�H�P�P�I�H�<�����������������������������������������z�z�n�n�z�����������z�z�z�z�z�z�z�z�z�z�������������������������������������������ĽĽнݽ����ݽнĽ����������ɺɺ��������ɺֺ�����������ֺɺɻ��-�6�:�S�_�l�p�x�������}�l�S�F�:�-� / m U L � /  Z ; / > F 2 Z S ( 9 � O  N g v J O i _ L  B L x W t - $  $ 6 q 8 - B K + n ? i I 6 � 4 Y N : v ( 1 Z T : B M H w u ] > w Q W    �  g  D  �  �    n  l  L    ,  �  X  +  �  _  �  �  4  �  �  h  �  �  �  �    	  �  D    �  �  �  ?    �  D  �     �    �  �  B  d  X  �  K  �  �  p    �    �  >  �  �    b  �  v  T  �  �  G  =  �  �  �  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@  @@    �    _  �  �  	3  	c  	�  	�  	�  	n  	  �    S  �  �  d  r  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ?  =  ;  9  8  6  4  )      �  �  �  �  �  �  �  �  t  b  )  e  m  x  �  �  y  [  :    �  �  z  E    �  �  K  �  �  �  �  �  �  �  �  �  �  �  �  |  w  w  w  s  j  a  &  �  �  .  p  p  k  b  U  D  .    �  �  |  5  �      �  �  Z  �  	  4  P  g  v  z  r  c  K  *    �  }  "  �  U  �  �  �  �  �  �  �  �  �  �      /  A  [  g  ~  �  �  �  !  O  $    I  O  �  �  �  �  �  �  �  �  �  �  �  y  ^  E  .    '  �  s  i  ^  Q  E  8  /  "    �  �  �  R    �  m  �  |  �  h  �  �  �  �  �  �  �  �  y  c  N  :  $    �  �  �  k  9    y  t  j  [  F  /      �     �  �  �  �  �  �  c  )  �               �  �  �  �  �  �  �  �  �  w  V  5     �   �  +  2  3  -      �  �  �  �  X    �  �  ,  �  !  �  {      �  �  p  -  �  �  _  #  �  �  �  t  K    �  �  J  �  S  �  �  �  �  �  �  p  L  %  �  �  �  `    �  "  u  �  �    �  
  2  7  2  &      �  �  �  �  �  u  c  [  \  f  �  %  @  D  G  K  N  Q  T  X  ^  j  u  �  �    {  w  a  G  .    5  2  /  -  *  '  $  ,  9  F  S  `  m  v  u  u  u  t  t  s  �  �    >  R  ^  c  c  [  I  .  	  �  �  b    �  !  �  a  �  �  �  �  �  �  �  �  �  �  u  T  1    �  �  g    �  [  T  J  ?  6  /  '        �  �  �  �  f  0  �  �  }  :  �  �  �  �  x  f  U  D  3  "      �  �      $    �  �  �  -        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  W  7    �  �  s  0  �  �  �  �  �  �  
            �  �  �  >    �  @  �  5  W  u  �  �  �  �  �  �  c  M  ;    �  �  �  �  O    �  f    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  k  �  �    '  ?  G  7      �  �  �  F  �  �  >  �    �  �  �  �  �  �  �  �  �  s  c  S  D  5      �  �  �  �  x  \  �  �    *    �  �  �  �  �  �  j  &  �  �  7  �  k  �    4  2  )       �  �  �  �  y  H    �  �  N    �  O  �  K  
  
]  
�  
�  
�  
�  
p  
R  
%  	�  	�  	%  �  ?  �    )  Q     �  �  �  �  �  u  m  o  p  W  9    �  �  �  �  �  �  n  <  	  �  �  {  m  ]  I  5     	  �  �  �  �  d  .  �  �  �  v  K  r  z  �  �  �  �  �  |  r  i  ^  T  I  >  3  '        �  �  �  �  �  �  �  �  l  K  $  �  �  �  n  1  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  Q  7      �  �  �  �  �  ^  W  B  @  �  �  �  �  P    �  �  ^    �  �  �  �    �  �  �  �  �  �  �  r  G    �  �  g  *  �  �  _     �  �          �  �  �  �  �  �  k  I  $  �  �  �  V    �  �  }  x  q  c  S  A  ,    �  �  �  �  ~  V  +  �  �  �  x  �  �  �  �  �  �  �  �  �  x  o  f  Z  I  ,    �  �  o  :  q  �  �  �  �  �    �  �  �  �  v  C    �  T  �  T    o  �  �  �  �  �  �  �  �  }  _  =    �  �  g     �  O  �  -  �  �  �  �  �  �  �  �  �  �  �  �    z  y  x  w  v  v  u  b  d  g  i  j  d  ]  W  P  G  =  4  +  "         �  �  �  �  �  �  n  Q  5  3  +    �  �  �  ^  I  0  �  �  H  �  y  	    �  �  �  �  y  V  /    �  �  ^    �  r  )  �  �  o  �  �  �  �  �  v  d  Q  =  '    �  �  �  �  W  (  �  �  �  �  �  �  �  �  �  �  u  k  a  X  N  D  7  $        �   �   �  �  l  d  X  D  /      �  �  �  �  �  d  4  �  �  	  |   �    )  &      �  �  �  �  ~  J    �  �  Z  )     �  �  �  ;  d  g  S  A  ;    
�  
�  
T  	�  	}  �  m  �  @  j  i  `  f    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  .    �  �  �  g  :    �  �  �  ]  0  �  �  �  D    �  �  ;  9  /  "       �  �  �  �  �  W  $  �  �  �  5  �  |         �  �  �  �  �  �  h  [  [  Q  ;  $    �  �  �  <  �  �  �  �  �  �    _  >    �  �  �  }  P  #  �  �  �  �  ^  �  �  c  1  �  �  �  �  �  X  �  �    u  �  
z  �  �  �  �  *  0       �  �  �  �  �  �  r  N  '  �  �  �  <  �  �  4  F  d  m  r  p  d  J  #  �  �  �  K    �  �  ?  �  �  {  e  �  �  �  �  x  l  _  R  D  7  )      �  �  �  �  �  �  �  :  ?  7  ,  #      �  �  �  �  �  w  J    �  �  G   �   ]  (      �  �  �  �  �  �  �  �  k  '  �  q    |  �  O  �  n  f  ]  U  M  E  <  4  ,  #    �  �  �  �  �  }  m  \  K  h  [  M  @  2  %      	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  g  Z  N  >  +      �  �  �  �  �  �    �  �  �  �  �  �  n  T  :    �  �  �  {  `  E  2      �  u  ^  F  +    �  �  �  x  J    �  �  e  !  �  �  C   �  �  �  �  �  �  x  I    �  �  ~  O    �  �  T    �    