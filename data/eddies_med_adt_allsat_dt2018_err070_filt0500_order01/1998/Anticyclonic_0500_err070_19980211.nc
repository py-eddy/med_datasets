CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�;dZ�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Mד   max       P��X      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       >t�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @F���
=q     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
>    max       @vo33334     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @O�           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @��@          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �e`B   max       >7K�      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B2�k      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�w�   max       B2��      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?2��   max       C�V      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?1�   max       C�Y�      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          z      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          )      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Mד   max       O�|      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�
=p��   max       ?Ә��@�      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �ě�   max       >t�      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��\)   max       @F���
=q     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�p��
>    max       @vo33334     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @O�           �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @� �          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�X�e+�   max       ?Ә��@�     �  V�               (                  _   C                     9         !      A   7   	   +   3         	      =   z               R   ,         1         &      2      
   (   
                  	      D   K   
   Nӥ�N�0�N��NV�BPop�Ol<OF8N(��O��N���P��XO�֩N��
N��N;�O��VN�gO%�P(�'N?eNgO�<N;.�P
s�Oǔ�N8�SO�E�O�NV�8N���N���N�z�O���Pc� N�$�N�a�N,�O^P'�O�|NN�F�N��wOΣmN؊6NsZO���O�	O�]�NdpN�RO^� N�"FNHO�N��N-4�N��Op�N[H�O��O�|O�2�N�4�Mד�ě��ě��e`B�49X���
��o��o:�o:�o<t�<49X<D��<D��<D��<D��<e`B<���<�1<�1<�1<�9X<�j<�j<ě�<���<���<���<�`B<�`B<�`B<�=+=\)=t�=#�
='�=,1=,1=49X=H�9=H�9=H�9=aG�=e`B=ix�=m�h=m�h=u=y�#=��=�+=�+=�7L=�hs=��w=���=��=��=�1=�^5=���>t�>t�CHNR[gptxutmg[XNCCCC##03<FD<;0)##*268CHC>6**########��������������������34B2.O���������tOB:33-.8<BJNU[_dghf[NB;3��������������������tt{��������ttttttttt��������������������z{~�����������zzzzzz��������������������������
����������	�����������:7<HKUWUTIH<::::::::��������������������##03U\bddaYO<0$#���������������������������������������413:Unz��������nHD<4������������������������������������������������������������')6ABDB>60+)''''''''������
#%>D@5#
����/<Han~}saUH</%)6BOTWa\O1)!���
/<JUTXYUH/#
�#/9:0/#������������������������������������������������������������74;<HTaz������zmaH;7��������464-!�������������������������������������������./27;HOHH@;/........a^]aht�����������xha����)5BNVN?5����)5<??=<:5)325668BNPTZZNB953333��������������������
)5BNMSXWKB5)zusv{{�����������{zz�������������������������9COWbh���������h\OC9�������
$*�������;9<EHJPRH<;;;;;;;;;;��������������������"/;HNTUUQKH;/"��
#$$#
 �������:;<DHLTTMH><::::::::


##003400#


���
���������LLO[ehih[OLLLLLLLLLLMO[`gt�������{tsg]VM�yxu����������������`aagmz~������zmiba``��������������������nuz�������������|unn���������������������������������������������������������������������������������Ľнݽ������������ݽҽнȽĽ����ĽĿy�~�������}�y�r�m�h�m�n�y�y�y�y�y�y�y�y�(�4�6�A�C�H�A�4�(�%�$�%�(�(�(�(�(�(�(�(�(�A�f������������Z�4����޽ܽ����(�n�{ŇŠŭŴŭŬŠśŔŇ�{�n�b�U�P�Z�b�n���������ûɻǻĻ����������x�p�m�o�x���������ûλлڻлû�������������������������������(�5�:�:�4�'�������߻߻������
���� ������޹�������g�����������������g�N�������(�A�g����������	�����������������������#�/�<�<�D�<�/�#�����#�#�#�#�#�#�#�#�zÇÎÓÓÕÓÉÇ�z�z�y�z�z�z�z�z�z�z�zÓàåáàÓÇÄÇÎÓÓÓÓÓÓÓÓÓÓ�r���������������������������r�b�Y�f�r�M�Q�Z�f�g�f�`�a�Z�V�M�A�4�3�4�8�A�L�M�M����������������������������ŵŶŹ�������5�A�N�Z�n�v�y�v�s�Z�N�I�=�,�#�����5�/�<�H�I�H�A�<�/�+�,�/�/�/�/�/�/�/�/�/�/����������������������������������������������(�5�*�(�!�������ݿѿſѿտ�ÓÕàÓÐÇ��z�o�zÇÏÓÓÓÓÓÓÓÓ��������������	�׾������������~�v����� �&�%�������������������������лܻ޻�����������ܻлȻλлллллм���4�M�Z�l�u�u�f�Y�M�4�'��������`�m����������y�m�`�T�G�;�3�,�.�4�C�G�`DIDVDbDgDmDbDVDIDIDGDIDIDIDIDIDIDIDIDIDI�zÇÓàìíñìàÝÓÇ�z�w�q�t�z�z�z�zE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��s���������������s�s�n�j�r�s�s�s�s�s�sāčĠĦĹľ��ĽķĳĦĚčā�t�h�c�i�tāĳĿ���
��#�&�"�������Ħč�t�t�~Ęħĳ������)�/�-�5�)�������������%�(�4�3�(�'���	������������"�#�/�;�C�?�;�/�.�"��"�"�"�"�"�"�"�"�"�F�S�_�l�t�r�_�S�K�F�D�:�-�+�-�6�8�:�<�F�������¿ǿɿ������y�m�h�T�L�C�D�R�_�y��Ƴ����������
�
���������ƳƧƟƧƢƩƳ�\�h�r�uƁƎƑƒƎƁ�u�h�`�`�\�X�\�\�\�\�ּּ��������ۼּʼʼȼʼּּּּ����-�@�H�K�H�B�)��������������������нݽ��������������ݽнʽϽно׾����	��	�	�����׾Ѿ;׾׾׾׾׾������������������������������������������������ʾ׾۾����׾ʾ����������������������������������������y�o�h�j�n�w��������
������������������������������G�T�W�`�b�`�`�T�G�;�2�1�;�?�G�G�G�G�G�G���
���#�(�+�-�(�#���
��������������ǡǬǬǪǣǡǔǒǈǃǈǋǔǚǡǡǡǡǡǡEEE"E*E,E*EEEE EEEEEEEEEE�������������������������v�}����������_�e�f�`�_�S�G�F�@�B�F�O�S�^�_�_�_�_�_�_�ݿ������ݿۿֿۿݿݿݿݿݿݿݿݿݿ��	���"�'�,�+�"��	�������������������	������������������������������������������������������������ŹŵŭŦŧŭŹ�����Ӽ������Լ�����������ʼ������������������ûĻ������x�l�_�Y�S�S�W�]�l�x�����a�d�n�u�n�m�a�U�H�<�H�I�U�`�a�a�a�a�a�a��#�&�'�#��
��
����������� 7 E T \ Z 9 ; p 3 F @ % . T R ( - 7 S S | A A _ 5 d = . 3 % ] E + 7 8 , X l " 8 A 3 ' E y , i k /   I 3 , ~ = - R : _  Q S    �  "  I  �    �  �  ]  &  �  �  �  �  K  J  ,  �  y  U  2  f  $  Y    �  p  �  �  e  �  �  �  �  �  �  �  g  �  �  ]  �  �  �  �  �    �  �  )  �  �  �  T  �  �  4  �  x  ,  {  �  �  ��`B�e`B�49X��`B=\)<�/<ě�<T��=o<�1=�
==��w<�t�<u<u=t�=o=<j=��-<���<���=e`B<�h=�9X=��
=\)=�O�=��w=49X=D��=�w=��=\> Ĝ=ix�=T��=@�=�%>   =�j=e`B=ix�=��=��P=}�=ě�=��=�G�=�o=��P=��=���=��=�1=�1=���=�;d=�Q�=�/>"��>7K�>�>�B�$B%V�B0#�B!�hB��B!�B"��BȂB �4BU�B�gBk�B �B,�By�B&V�B�B#bB�}BBh�B��B�7BR*B��B9�B�WByB�PB".xBc�Bg�A��QB��BO�B+��A��wBT�B��Bw�BօB!�B��B)B�Bi�Bo�B2�kB,�IB+�B!�5A��B�LB>eB%�B]pB0}B	�wB
W$A��B+^B�SB��B��B	.B%�*B0E�B!ڱB:rB@B"��B�hB!@B>eB�QB?/B�_B`B~|B%��B�|B?cB�?B��B.�B�B�&BVB�~BL�B��B\�B��B"?JBJBtA��4BØB@]B+��A��7B��B�0BF0B/�B �"B�yB)@�B��B�B2��B,��B�B!�&A�w�B�hB@lB%2�B�9B?�B	ѸB
;�A�~�B?�B�VB�B�HA��A*�Al��A8�A7�GA�@��@�~�@�A4?2��A�	�A�8[A�<A�z�A�q�@�tA=0A�N{A�A�8A�[A���Aɪ:APo�Aҁ0@�^�@��QAhQjC�|7A�g�C�VAE#A� A��A�B�A�c�A���@�/�Ao)�B��B��Ai�AԊ�A-߰AV�yA�̠AOEA A0SOAetA�VB�LC�n�AH��@���A}��A�wA�%�A���@�%�@�A��A���A��cA+�Al�A8�?A6�A�A@�ͯ@��z@��p?1�A���A�hA�x�Aɀ:A� ;@网A=,aA��A��`A,A�v$A��zAɾjAO�/A��@���@�Ah�*C�x A�jOC�Y�AD�A�zA�<A�|�A�|A��h@���Ap�;B�B��A��A�m�A/	AUj(A��RAP�LA�RA0�AeG�A��B�C�i�AH�\@���A}�A��A�#A�}FA&@���A�,�A���               )                    _   C                     9         "      A   7   
   ,   3         
      =   z               R   ,      	   1         '      3      
   )                     	      E   L   
                  9                  9                        +               -   !      #   !               !   5               +                           '                                    )   #                                                               #                                                            !                                                               )   #      Nӥ�N�.�N��NV�BO$K�OROO�DN(��Oa̠N���O�׆Ov��N��
N��N;�O�elN�c�O��O��`N?eNgO7$�N;.�O�5�N�N8�SO�g=O��uNV�8N�N���N�z�OU<�OdY�Na��N�a�N,�N�d�O�:?Ou�7N�F�N��wO+N؊6NsZO���O�	O�eNdpN�RO^� N�"FNHO�N��N-4�N��Oc�N[H�O��O�|O�2�N�4�Mד  �  �  �  �  6  �  &  �  /  A  �  	�    �    3  �  <  h  I    �  �  �  �    �  v  �  �  �  d  	�  �  e    �  ;  	�    �  a  4  ^  �  }    ,  �  �  	�  �  F  �  �  7  k      
�  �    Ҽě���9X�e`B�49X<��
�o;�o:�o;ě�<t�=y�#<�h<D��<D��<D��<u<��
<�j=�P<�1<�9X<�h<�j=49X=ix�<���<�/=,1<�`B=C�<�=+=e`B=���=8Q�='�=,1=8Q�=�7L=aG�=H�9=H�9=ix�=e`B=ix�=m�h=m�h=�1=y�#=��=�+=�+=�7L=�hs=��w=���=���=��=�1=�^5=���>t�>t�CHNR[gptxutmg[XNCCCC#0<<><70&##*268CHC>6**########��������������������ZTTW[`ghjtwz}���th[Z4//5;BNT[^cffd[VNB<4��������������������tt{��������ttttttttt��������������������z{~�����������zzzzzz����������������������������������������	�����������:7<HKUWUTIH<::::::::�������������������� $0IU[bcc_XN<0%$����������������������������������������:<HXaz��������naUH=:������������������������������������������������������������')6ABDB>60+)''''''''���� 
#058:83/#
��-/2<HPUYUMH<0/------)6BOSV\a\OB/)		#/<BHOQQHF</#	#/9:0/#������������������������������������������������������������FGJT[amz����|zmaVTJF��������������������������������������������������./27;HOHH@;/........c__bhtx�������thcccc�����)0562)���

)359<<;85)
325668BNPTZZNB953333��������������������)5ILRVWVIB5)zusv{{�����������{zz�������������������������9COWbh���������h\OC9��������������������;9<EHJPRH<;;;;;;;;;;��������������������"/;HNTUUQKH;/"��
#$$#
 �������:;<DHLTTMH><::::::::


##003400#


���
���������LLO[ehih[OLLLLLLLLLLNP[agt���������tg^WN�yxu����������������`aagmz~������zmiba``��������������������nuz�������������|unn���������������������������������������������������������������������������������Ľнݽ�������ݽڽн˽Ľ½��ĽĽĽĿy�~�������}�y�r�m�h�m�n�y�y�y�y�y�y�y�y�(�4�6�A�C�H�A�4�(�%�$�%�(�(�(�(�(�(�(�(���(�4�A�M�Z�]�Z�U�M�A�4��������n�{ŇŔŠŬũŠřŔŇ�{�n�b�W�U�R�[�b�n���������������������������}�x�v�x�|���������ûλлڻлû�����������������������������'�0�7�6�4�'������������������
���� ������޹�������s�������������������s�g�Z�R�J�L�P�Z�g�s������	�����������������������������#�/�<�<�D�<�/�#�����#�#�#�#�#�#�#�#�zÇÎÓÓÕÓÉÇ�z�z�y�z�z�z�z�z�z�z�zÓàåáàÓÇÄÇÎÓÓÓÓÓÓÓÓÓÓ���������������������������r�c�Z�f�r��A�M�Z�f�_�Z�M�A�4�4�4�9�A�A�A�A�A�A�A�A������������������������ŹŹŸŹ���������A�N�Z�g�k�q�m�g�Z�N�A�9�-�$����#�5�A�/�<�H�I�H�A�<�/�+�,�/�/�/�/�/�/�/�/�/�/��������������������������������������������$����������ݿֿѿϿѿݿ��ÓÕàÓÐÇ��z�o�zÇÏÓÓÓÓÓÓÓÓ�����ʾپ��������׾ʾ������������������������������������������������������лܻ޻�����������ܻлȻλлллллм�4�@�M�Y�k�t�t�f�Y�M�@�4�'�������T�`�m�y�}������y�m�`�T�G�<�;�7�6�:�D�TDIDVDbDgDmDbDVDIDIDGDIDIDIDIDIDIDIDIDIDIÇÓàãìíìà×ÓÇÄ�z�y�z�~ÇÇÇÇE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��s���������������s�s�n�j�r�s�s�s�s�s�sčĚĦİĳĵĵıħĦĚčā�}�q�o�t�vāč������������
�������������ĿĻĸĽ��������(�)�*�*�-�)�������������%�(�4�3�(�'���	������������"�#�/�;�C�?�;�/�.�"��"�"�"�"�"�"�"�"�"�F�S�_�l�r�p�l�`�_�S�H�F�<�=�=�?�F�F�F�F�y���������������������m�`�W�Q�R�X�a�m�y����������������������������ƳƬƩư���\�h�r�uƁƎƑƒƎƁ�u�h�`�`�\�X�\�\�\�\�ּּ��������ۼּʼʼȼʼּּּּ����+�>�G�I�B�)���������������������нݽ��������������ݽнʽϽно׾����	��	�	�����׾Ѿ;׾׾׾׾׾������������������������������������������������ʾ׾۾����׾ʾ����������������������������������������|�y�w�t�x�y��������
������������������������������G�T�W�`�b�`�`�T�G�;�2�1�;�?�G�G�G�G�G�G���
���#�(�+�-�(�#���
��������������ǡǬǬǪǣǡǔǒǈǃǈǋǔǚǡǡǡǡǡǡEEE"E*E,E*EEEE EEEEEEEEEE�������������������������v�}����������_�e�f�`�_�S�G�F�@�B�F�O�S�^�_�_�_�_�_�_�ݿ������ݿۿֿۿݿݿݿݿݿݿݿݿݿ��	���"�'�+�*�"��	�������������������	������������������������������������������������������������ŹŵŭŦŧŭŹ�����Ӽ������Լ�����������ʼ������������������ûĻ������x�l�_�Y�S�S�W�]�l�x�����a�d�n�u�n�m�a�U�H�<�H�I�U�`�a�a�a�a�a�a��#�&�'�#��
��
����������� 7 F T \ B 1 2 p - F "  . T R ) 8 2 G S | / A ( / d , 0 3 . ] E  ) ; , X K  5 A 3 % E y , i U /   I 3 , ~ = ' R : _  Q S    �  �  I  �  �  �  !  ]  �  �  V  �  �  K  J    �  A    2  f  �  Y  $  �  p  O    e  �  �  �  �  �  w  �  g  �  �  �  �  �  �  �  �    �  �  )  �  �  �  T  �  �  4  �  x  ,  {  �  �    D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  �  �  �  �  �  �  v  _  G  +    �  �  �  �  `  =      f  �  �  �  �  �  �  �  �  �  �    j  L  (  �  �  �  G  /    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  w  q  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  j  V  C  /    �  �     +  (        '  #    )  6  &    �  �  p  g  g  �  �  �  �  �  h  D    �  �  �  �  t  R    �  �  G  �  �          %  !             �  �  �  ~  O    �  �  '  �  �  �  �  �  �  �  �  �  �  �  {  _  A  "     �  �  �  r      *  /  +  $       �  �  w  E    �  �  �  c  �  Q  �  A  @  7  "    �  �  �  �  �  v  U  *  �  �  �  L    �  s    h  �  �    F  c  �  �  �  �  �  �  J  �  '  O  .  �    �  	A  	�  	�  	�  	�  	�  	�  	�  	h  	1  �  y    {  �    )  �  �    �  �  �  �  �  �  �  �  �  �  i  J  ,    �  �    F  x  �  �  �  �  �  �  �  �  �  �  �  �  �  }  o  a  R  D  5  '          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  .  3  1  -  '      �  �  �  �  _  &  �  �  c  D  %    �    �  �  �  x  m  a  P  >  *    �  �  �  ~  P    �  w  �    8  8  %  	  �  �  �  x  K    �  �  q  ?    H  �  �    i  �  
  ;  W  e  f  X  >    �  �  .  �  8  �  5  �    I  I  B  ;  4  -  %          �  �  �  �  �  �  �    "  B    	    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  P    �  �  ]  3  �  �  !  G  =  "  �  �  �  �  �  �  �  }  n  _  N  =  +      �  �  �  �  �  �  �  ;  y  �  �  �  �  �  o  &  �  h  �  {  
  �  �    Z  �  �  -  ^  �  �  �  �  �  �  �  �  �  �  P  �  -  �  ,  �      �  �  �  �  q  G    �  �  �  u  M  %  �  �  �  �  �  *  v  \  B  "  �  �  �  e  '  �  �  T  �  �  I  �  �    f  �    A  ^  m  u  s  g  N  +  �  �  ~    �  �  [  �    �  �  �  v  c  N  5    �  �  �  z  Q  '  �  �  �  j  '  �  e  t  �  �  �  �  �  �  �  �  �  `  (  �  �  b    �  k  �  s  �  �  �    m  Y  D  .        �  �  �  �  �  �  �  �  �  d  W  J  >  0       �  �  �  �  �  �  �  p  V  3     �   �  	  	V  	|  	�  	�  	�  	�  	�  	�  	�  	�  	G  	  �  P  �  �        	�  	�  
  
  
  
_  
�  .  �  �  �  �  �  �  ?  
�  	�  @  �  e  3  9  A  K  T  ^  a  H    �  �  p  <    �  �  y  ~  �  �       �  �  �  �  �  �  �  �  p  ^  K  3       �  �  �  z  �  �  �  �  �  �  s  c  P  6      �  �  �  �  `  =     �  �  �  	  8  .      �  �  �  �  h  1  �  �  I  �  �  �  �  �  	W  	�  	�  	�  	�  	�  	�  	�  	�  	\  	  �  '  �    d  �  �  �  �  �       �  �  �  �  �  X    �  �  &  �  )  �  �  �   �  �  �  �  �  �  �  �  �  |  u  n  g  ]  S  F  6  '  /  @  Q  a  K  6  "    �  �  �  �  �  �  m  T  5    �  z  #   �   r  /  4  .       �  �  �  V  +  �  �  z  -  �  f  �      �  ^  Z  X  O  M  D  0       �  �  �  �  z  f  C  
  �  �  :  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  p  ^  @       �  �  �  h  3  �  �  F  �  c  �  .    �    �  �  �  �  �  �  �  �  �  �  �  p  ]  B    �  �  �  �  �  �      $  )  (  $  '  +  +  "    �  �  *  �    v  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  x  s  n  i  e  `  �  �    i  R  :  "  
  �  �  �  �  �  p  P  -    �  �  �  	�  	4  	  �  �  �  �  k  ;    �  �  7  �  ]  �    P  �  �  �  {  o  a  S  E  6  $    �  �  �  �  �  q  S  6    .  H  F  +    �  �  �  S    �  �  c  "  �  �  ]    �  �  H  �  �  �  �  �  |  j  Z  I  8  &    �  �  �  �  �  p  $  �  p  �  �  �  �  �  y  c  M  8  "    �  �  �  �  �  �  �  s  T  7  *        �  �  �  �  �  �  �  o  V  =     �   �   �   �  i  f  V  H  8  $    �  �  �  L    �  �  q  6  �  �  $  B      �  �  �  �  �  �  t  Y  8    �  �  �  Q    �  Z  �    �  �  �  �  {  Z  5    �  �  �  �  _  :    �  �  I  �  
�  
�  
�  
�  
Z  
  	�  	�  	+  �  b  	  	&  �  "  b  y  o  �  t  �  �  �  �  �  �  d  .  
�  
n  	�  	~  �  b  �    &  �  �  s    �  �  �  �  t  S  3    �  �  �  �  u  V  <  5  0  :  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  t  m  e  ^