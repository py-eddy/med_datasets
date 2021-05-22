CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��l�C��       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�]�   max       P�u>       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       ;�`B       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?=p��
>   max       @F��Q�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��Q��    max       @vxQ��     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q�           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�F        max       @�v            7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �t�   max       ��o       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B5AM       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�x�   max       B55�       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >\I�   max       C�E       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�mu   max       C��k       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          m       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          5       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�]�   max       Py0�       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��3���   max       ?�'�/�W       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       ;�`B       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?nz�G�   max       @F��Q�     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���
=p�    max       @vw��Q�     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @Q�           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�F        max       @�O�           Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��ߤ?�   max       ?�'�/�W     �  ]         &      7         J                  ,            
      %   `   #                                    O         
      !      m   O   &            6                              T   	            '                        O J�OS�PVT�N���P/M�]�Nį�P�BN�f}O�	N��XO���N�kP
��NƦ�O#/N͖RO?�O�vCP)��P^�P_�N�B�Ny&CN� N���O��@O&��N �mN�
	Pl�N���O�D�P�u>Nu�vO��O(1�NQ�"O�DOS��P-��O�L�O��kNЖ�NN��O5(7O�b&N��fNWPhO��O���N5�?N�_N�N�i�NR �O��N���Oh�yO�Np hO�c�N���O��N<=�N%7CN���N���N��O �;�`B;�o��o�o�o�o�D���D���D���D�����
���
��`B�o�o�t��#�
�49X�D���D���D���e`B�u�u��o��t����
��1��j�ě������������o�o�\)�t��t��t��t��t��t���P��w��w�,1�<j�<j�<j�@��@��@��@��@��P�`�P�`�P�`�T���Y��]/�aG��ixսixսm�h�q����o��+��7L��C������������������[ctz�������������za[?P|�������zqaTHB;54?16BCJOOQROB>63.-1111O[t������������hOIGOEIU\bdbbURIEEEEEEEEE)5<<;54)%�������#2985+
����GHRUZahllaUHAAGGGGGGyz����������������zy�����������������������
#-31+#
�������������������������
#<HUanxnaZN/ ���NOVX[hnqomih[OMHNNNN�������������������������������������������������GHTaqz�����|�zmTKG������������ ��������������

��������DHKRTmz������xvmTHBD������������������������������������������������������������ 
 �    ��������������������}�������������|xuvy}')6BDCB6)'''''''''''��
#//1/##
 ����������������z}�����9BNT[cc\[NLB>9999999#<IlzwynmbUI<2--)# #}����������������xw}�����������������������
#0<@=<.#
�������������	����������������������������ft���������������jef�����������������������������������������������������)*08<A5)��������������������������������������������:<FIUWabfklbU?<9778:lqz�����������znfddl����������������������������������������),5BN[`[VPNIB54+)!q~�������������wthmq`ahnz�~znia[````````��������������������
"#










���������������������
������������#/HSVVZ\YUH</#��������������������NT[fhjpt����{th[WQLN)-58BCEDB55)LOQ[ahknlh`[OMLLLLLLXct�������������tgZX��������������������py{�������������|qopaht~���th]aaaaaaaaaa��������������������� '%���������  #!
1<CEFHIJHF<932451111\\ghihffa[ODBA>?BHO\�w�t�q�t ¦°­¦¥�U�H�<�4�+�/�=�>�H�U�X�_�_�a�n�z�n�j�[�U����������5�A�m�x�������������Z�5��F�>�F�F�S�_�_�l�x�����x�p�l�_�S�F�F�F�F�������v�p�r�t�x�����ûл޻��ڻƻ������ʼƼ��������ʼʼӼּּܼʼʼʼʼʼʼʼ��������)�5�A�B�E�B�5�0�)� �����������g�N�;�5�+�)�5�A�Z�g����������������������(�5�?�?�<�5�(������������������������������	������������������#�,�/�<�H�Q�R�H�<�<�/�#����Å�z�y�z�Îàìù��������������úìàÅ����ùõù������������������������������������ ������5�A�O�[�q�s�[�B�6��.�+�"���
��"�.�;�G�H�L�I�G�;�.�.�.�.��
�	�����	�	��"�'�/�3�7�7�1�/�"�ìêààÞàäìù����������ÿùìììì�����	���(�5�A�C�N�X�W�X�N�A�5�(���ƳƔƅ�u�h�uƁƚƧ����������������������������Ƴƪƻ������0�B�F�C�@�0��'�����ܻѻϻܼ���4�Y�r�������Y�'�H�;���������� ��;�H�R�W�a�m�u�y�o�T�H������|�������������������������������D�D�D�D�D�EEEEEE	ED�D�D�D�D�D�D�D��U�R�H�?�<�2�0�<�H�U�X�a�a�e�a�]�U�U�U�U�f�]�Z�N�Z�a�f�s�������������s�f�f�f�f��������ŪşŠŭŹ��������*�7�@�6�*����������������ѿݿ�����������ݿѿĿ����#����#�/�0�8�1�/�#�#�#�#�#�#�#�#�#�#�t�n�g�f�f�g�h�t�v�{�t�tĳĭįĺĸĿ��������G�S�Q�<�#�����Ŀĳ�������������	������	��������������»������л����'�/�*� �������ܻлºe�Y�:�A�L�l����������!�-�C�9�-���ֺ��e�����������������Ⱦʾ˾ʾǾ��������������0�$� ��
����$�0�=�>�I�N�O�P�N�I�=�0�y�m�`�X�T�M�T�`�f�v�y�����������������y��
����"�.�;�A�;�.�"���������G���'�,�0�.�;�G�T�Y�`�h�v�~��x�`�T�G�I�@�?�C�I�M�S�V�b�o�w�{ǂǂ�{�x�o�b�V�I�黷�����~���������ûܻ�����������M�@�4�'���%�7�M�Y�f�r�����������r�Y�M�Y�R�S�d�h�tāčĚĤĥĦĥĜĚčā�t�h�Y����������������������������������������û��ûͻлܻ����������ܻлûûûþ�����������(�4�8�=�M�Q�M�A�4���������������������Ϲܹ�����	����ܹ����������������������������������������������������������������������������������������������������������ŭŠŞŘőŔŠŭŹ������������������Źŭ�H�E�;�6�6�;�H�I�P�T�`�T�H�H�H�H�H�H�H�H������!�.�3�.�"�!����������Ŀ����Ŀѿݿ޿ݿܿѿĿĿĿĿĿĿĿĿĿ�Ç�}ÂÇÓàëãàÓÇÇÇÇÇÇÇÇÇÇDbDaDVDSDVDWDbDoDpDvDpDoDbDbDbDbDbDbDbDbE�E�E�E�E�E�E�E�E�E�FFFFF FFE�E�Eٺ~�r�r�e�c�e�k�j�r�~�����������������~�~�ù��������������ùϹܹ��� �������Ϲ�ùøîìäì÷ù��������������������ùù�Ľ����������Ľнݽ�ݽٽҽнĽĽĽĽĽ�ĳĦĞĚěĤĳ�������������
��	������ĳ����������������������������������������#�
���������������	��#�<�H�R�U�H�<�/�#��ݼڼ���������������������������	������������������������5�-�)�2�5�A�N�W�U�P�N�A�5�5�5�5�5�5�5�5�l�a�`�l�y���������������������y�l�l�l�lD�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��f�g�s�y�������������������������x�s�p�f  d V W 6 w K > " D f * t X K S / 5 b A c I : f E 5 � E 4 > [ 9 * P : H / W 7 . ( 8 2 M ~ L S J o : ( t j $ ? A   . 2 ) :   : I C 2 0 f  0    �    �  t  A  �  �  �  =  �  6  =  �  �  �  �  �  �    �  z    �  �  �  P  �  C  "    �  �  0    =  x      �    �  �    �  �  �  �  �  K  8  b  N  6  �  k  �  �  �      2  �  �  e  @  �    �  �49X�ě���w�e`B�e`B��o�o���-�49X�t����
�o�49X�T���T����9X���ͼ�1�C��H�9��/�D����1��P�������8Q��P��`B�'aG��C��Y���G���P��o�<j�,1��C��L�;t���xս��P�]/�0 Ž�o�����Y��Y��u��hs�H�9�Y��Y��ixս����	7L�u���T�}�q���ě������9X��+��C����T������T��hBuB��A��B�BƣB'/B�Bs~B�nBI�B��B�B �
B�IBk�B B�B��B[[A�* B"�B �1A��BF�B;B!'B0BgB
��B��B+ABg�B@�B&�>B(�B5AMB+_B;�BAqB
�BF�B�B�B(!Bv�B*&�B&�7B��BWBʝB�B
��B�MB�DB{)B�uB�BK�B!9B�|B�BP�B
n�B��B
��BMB"�rBC�Bg�B�4B�BA�B��A�s�B�$BD�B&�B��BDqB��B�B��B=�B ��B¯B9B H7B�ZB��A�x�B�XB ��A�~!Bf�B?�B!#�B9�B�B
�3B�B�QB@BCYB&ɴBxLB55�B@$B�aBB�B
�$B��B®B@!B:xB�B*�B&��B��BCzBŏBE�B
;B��B�zBB%B�NB��B��B ѭB�hB7�B|�B
A�B��B
��B�lB"�BH~B@B��B��A�܊A�3A���@�Sc@�<u@�z�A�XnA���A���A��A�C�A�6=A�)A��AaKA��LA�@�A�MB�~B�L@٘�A���AJ	PC�P�A��aAB�RA�b�A{kA���A���A�A��@�;�@+cgAM�8B
G�Al�&A_��AfU�B0`@�u�@�{0A�m�A��@��A5�$>\I�A���A��&A�8�A��~A���A�WAzWZA�ynC���C�E@30>� �AΘ�A(�A�ܙA匄A�@�A`�@XTA�k�A?-C��wA�e�A�y�AĀ~A��f@�N@���@�J�A�ޝA��]A�leA�}UA�yA�k�AΊ�A׆&Ab�?A���A� A�s�B��B	�_@�cA���AJ*C�W Aě�AC��A��lA|O�A�z�A�tXA�}cA�m�@���@�vAN^OB
?�An�A^6�Af��BAj@�w�@��LA�Q+A��`@�49A5=�muA�i�A�[hA�d�A�QA�J?A�
Ay�A�9JC��wC��k@e>R�FA΍qA(ZA��A停A�A��@X� A�|MA�iC��5A�q�         '      7         K   	               -                  &   a   #            	                  	      O               !      m   O   &            6                              T   	      	      (   	                              1      %         '                  -               )   +   9   '               +            )      %   =               %      -   #               !                              !               %      #                           !      %                                          )   '                     +                  %   5                     %                                                                                       O J�N��,O�arN���OڈM�]�Nį�O�z�N�f}N�+�N��TO���N�kO�H*NƦ�O+)N�x�O#�BO�vCP8eO$e'O��!N�B�N`HNq��N���O��@O9N �mN���O�ωN��EO�4(Py0�Nu�vOV��O(1�NQ�"OfP�O?MO�d�O���O��kN��NN��O�	Ov�NI��NWPhO��O�?�N5�?N�_N�N�i�NR �O�Y4N���OY��O�Np hOǦ8N���Oe�fN<=�N%7CN���N���N��O �  0  �  z    �  �  �  X  �  	  �    d  R  �  �  p  �  M  4  	0    �  	    &  '    �  �  �  ?       D  `  ~  �  �  j  �     �    �  R  b  �  �    �  5  Z  ~  \  �    �    �  �  �  a  �  �  �  u  v  :  a;�`B�ě�����o�#�
�o�D�����D�����
�o���
��`B��1�o�49X�D���D���D����C���o��1�u��o���㼛�㼣�
��9X��j�����C���/�o���o�#�
�t��t��<j��P�m�h�0 Ž�P�'�w�49X�m�h�@��<j�@��H�9�@��@��@��P�`�P�`��C��T���]/�]/�aG��}�ixս�o�q����o��+��7L��C������������������mnz������������znfmmJP]mz}�����zmaTRJHGJ16BCJOOQROB>63.-1111LO_t������������hRMLEIU\bdbbURIEEEEEEEEE)5<<;54)%���
! 
�������GHRUZahllaUHAAGGGGGG|�������������||||||�����������������������
#-31+#
������������������������	!/<IQPLC</#
 � 	NOVX[hnqomih[OMHNNNN�������������������������������������������������GHTaqz�����|�zmTKG����������������������������������������Tamz�����xfaTMHJMNPT������������������������������������������������������������ �    ���������������������������������}yvwz')6BDCB6)'''''''''''��
#-/0/#
�������������������������;BGNP[bb[YNMB?;;;;;;"'<IessnnkhbUI<5/+%"{����������������zy{�����������������������
)/8=9/(#
�������������	����������������������������t���������������tnnt�����������������������������������������������������)*08<A5)��������������������������������������������;<AIQUY_bchdbU<9899;oqwz���������zpnkjko����������������������������������������),5BN[`[VPNIB54+)!ot�������������ztmno`ahnz�~znia[````````��������������������
"#










���������������������
������������
#/<HQUWWQH</#	
��������������������NT[^hnt���~zth[WROLN)-58BCEDB55)LOQ[ahknlh`[OMLLLLLLZ[gt������������tf]Z��������������������suy��������������wsaht~���th]aaaaaaaaaa��������������������� '%���������  #!
1<CEFHIJHF<932451111\\ghihffa[ODBA>?BHO\�w�t�q�t ¦°­¦¥�H�H�>�<�:�6�<�D�F�H�N�U�X�Y�^�_�Z�U�H�H�5�(������(�5�A�N�Z�g�x�����s�g�N�5�F�>�F�F�S�_�_�l�x�����x�p�l�_�S�F�F�F�F���������{�u�v�z���������лֻڻ�ջû����ʼƼ��������ʼʼӼּּܼʼʼʼʼʼʼʼ��������)�5�A�B�E�B�5�0�)� �����Z�R�G�C�G�N�Z�g�s�������������������g�Z��������(�5�?�?�<�5�(���������������������������������������������������#�/�<�H�K�H�<�6�/�#������Å�z�y�z�Îàìù��������������úìàÅ����ùõù����������������������������������#�'�)�6�B�O�[�a�g�_�`�W�O�6�)���.�+�"���
��"�.�;�G�H�L�I�G�;�.�.�.�.���	�����	��"�$�/�0�4�3�/�+�"��ììáàààæìù����������ûùìììì��������(�5�<�A�N�R�Q�R�N�A�5�(���ƳƔƅ�u�h�uƁƚƧ������������������$���������ƿ����������0�>�C�@�?�<�0�$�Y�U�M�@�?�4�3�4�@�M�Y�_�f�r�s�}�~�r�f�Y���������/�;�H�N�T�a�g�j�a�T�H�;�"��������|�������������������������������D�D�D�D�E EEEEEEED�D�D�D�D�D�D�D��H�A�<�6�9�<�H�U�U�^�`�U�H�H�H�H�H�H�H�H�f�_�Z�U�Z�f�l�s������������s�f�f�f�f��������ŪşŠŭŹ��������*�7�@�6�*����������������ѿݿ�����������ݿѿĿ����#����#�/�0�8�1�/�#�#�#�#�#�#�#�#�#�#�t�r�i�i�j�t�|�v�t�t�t�t�#��
�����������������
��0�<�I�I�<�0�#���������������	������	�������������лû������л�����'�,�(������ܻк~�e�S�U�f�o�������ֺ��!�+�)�����ֺ��~�����������������Ⱦʾ˾ʾǾ��������������0�$�������$�0�=�E�I�L�L�M�I�@�=�0�y�m�`�X�T�M�T�`�f�v�y�����������������y��
����"�.�;�A�;�.�"���������6�,�.�/�0�5�;�G�N�T�`�j�m�t�u�o�`�T�G�6�I�D�B�F�I�O�V�b�o�v�{ǁǀǀ�{�v�o�b�V�I�л��������������ûл������������мY�@�4�'���)�@�M�Y�f�r������������f�Y�Y�R�S�d�h�tāčĚĤĥĦĥĜĚčā�t�h�Y����������������������������������������û��ûͻлܻ����������ܻлûûûþ�
������������(�4�5�;�M�A�4�(�����������������������ùϹ�������ܹϹ����������������������������������������������������������������������������������������������������������ŹŭŠśœŗŠŭŹ��������������������Ź�H�E�;�6�6�;�H�I�P�T�`�T�H�H�H�H�H�H�H�H������!�.�3�.�"�!����������Ŀ����Ŀѿݿ޿ݿܿѿĿĿĿĿĿĿĿĿĿ�Ç�}ÂÇÓàëãàÓÇÇÇÇÇÇÇÇÇÇDbDaDVDSDVDWDbDoDpDvDpDoDbDbDbDbDbDbDbDbE�E�E�E�E�E�E�E�E�E�FFFFFFE�E�E�Eٺ~�r�r�e�c�e�k�j�r�~�����������������~�~�ù������������ùϹܹ�����������ݹϹ�ùøîìäì÷ù��������������������ùù�Ľ����������Ľнݽ�ݽٽҽнĽĽĽĽĽ�ĿĳĦĠğĠĦĳ�������������������Ŀ����������������������������������������/�#��
������������
��#�/�<�K�Q�H�<�/��ݼڼ���������������������������	������������������������5�-�)�2�5�A�N�W�U�P�N�A�5�5�5�5�5�5�5�5�l�a�`�l�y���������������������y�l�l�l�lD�D|D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��f�g�s�y�������������������������x�s�p�f  m L W 7 w K 7 " W ` * t < K L 2 3 b : * A : ` D 1 � @ 4 G I 0 2 ? : ? / W   , ( 3 2 @ ~ D P < o : % t j $ ? A  . ' ) :  : B C 2 0 f  0    M  �  �  �  A  �  "  �  �  �  6  =  5  �  I  �  a  �  t  U  �    �  |  �  P  i  C    w  �  �  .    �  x    �  �  :  �  �  �  �  S  )  m  �  K    b  N  6  �  k  `  �  �      �  �  �  e  @  �    �    @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  0  -  $      �  �  �  �  i  D    �  �  y  S  P  �  �  D  u  �    ;  l  �  �  �  �  �  {  W  '  �  �  �  �  \  +  �  ~  �  �  �  $  E  X  e  l  z  e  ;    �  �  p  )  �      �      �  �  �  �  �  �  �  �  �  ]  1    �  �  �  c  @    �  �  �  �  �  �  �  F    �  �  �  {  [  "  �  i  �  �  +  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    |  y  w  t  �  �  �  �  �  �  �  �  �  z  o  d  a  b  d  e  w  �  �  �  �  =  �  �  �  #  I  V  W  K  .  �  �  F  �  )  �    �  �  �  �  �  �  �  �  �  �  x  l  _  N  >  *      �  �  :  |          	  	  	    �  �  �  �  �  �  w  ^  G  :  -  !  �  �  �  �  �  �  �  �  |  j  S  0    �  w  L  �  D  �  T    |  p  U  +  �  �  {  2  �  �  �  z  [  I    �  �  �  :  d  U  F  8  )        �  �  �  �  �  o  O  6      �  �  :  ;  �  �  5  M  P  E  ,    �  ~  #  �  �  2  �  j  �  �  �  �  �  �  �  ~  z  u  p  m  j  g  X  B  -     �   �   �   �  �  �  �  �  �  �  �  z  a  B    �  �  �  s  H    �  �  �  b  j  o  o  l  f  Z  K  7      �  �  �  t  L  .    �  �  �  �  �  �  �  �  �  {  n  _  M  9  #    �  �  �  �  �  �  M  B  /      �  �  �  �  �  �  �  |  R    �  �    p   �  �    .  3  ,         �  �  �  T    �  q    �  e    �  �      9  v  �  \  �  �  	  	/  	(  �  �  _  �  8  J  �  W  �  �              �  �  �  �  a  .     �  �    R    �  �  �  �  �  �  �  �  �  �  z  k  ]  M  =  ,    �  �  e  		  	  �  �  �  �  W  (  �  �  h  "  �  �  s    l  �    j  �  �              �  �  �  �  �  ]  9    �  �  �  
      #  &  $  !         �  �  �  �  �  �  x  U    �    '    �  �  �  �  �  �  �  ]  C  !  �  �  J  �  �  p    l  �      �  �  �  �  �  �  �  �  p  I    �  �  K    �  �  �  �  v  l  b  W  M  B  1    �  �  �  �  �  f  B     �   �  �  �  �  �  �  �  n  [  L  A  A  F  F  E  G  J  R  j  �  �  =  H  C  7  n  �  �  {  k  O  )  �  �  m    �  i  2  @  y  !  .  :  :  2  )      �  �  �  �  �  �  j  >    �  L   �            �  �  �  �  �  �  �  �  �  d  2  �  �  �  �  �         �  �  �  �    V  (  �  �  F  �  #  ?  6  d  N  D  8  ,         �  �  �  �  �  �  �  �  �  }  `  A  !    8  N  Z  _  ^  R  8    �  �  e    �  u    �  3  �  �   �  ~  y  s  e  U  @  )    �  �  �  �  e  ?    �  �  �  ~  �  �  �  �  �  s  S  3    �  �  �  �  _  8    �  �  �  t  P  .  k  �  �  �  �  �  �  �  �  �  �  s  >  �  �  X  �  [  �  i  j  g  c  a  R  <  "    �  �  �  O  %      �  �  \     #  �  �  �  �  �  �  {  E  
�  
�  
-  	�  	-  �  �  �    U  V  
�        
�  
�  
�  
f  
   	�  	|  	  �  4  �    o  ?  �  �  �  �  �  �  �  �  |  _  <    �  �  �  6  �  ^  �  T  �  w    
              
  �  �  l  2    ;  �  �  d    �  �  �  �  �  �  �  �  �  �  �  �  }  z  w  t  i  ]  P  C  6  B  N  Q  K  B  8  .  *  $    �  �  �  �  S    �  �  9  �  �    <  S  a  _  N  *  �  �  �  Y    �  �  @  �  l  �  
  �  �  �  �  �  �  �  �  �  �  z  j  S  ;    �  �  �  M    �  �  x  g  \  Q  ?  (    �  �  �  �  �  �  �  �  �  �  2      
  �  �  �  �  �  �  �  �  �  ^  (  �  �  �  �  �  �  �  �  �  �  �  �  �  v  Z  <       �  �  �  F  �  �  F  B  5  1  -  )  &  "            �  �  �  �  �  �  �  �  �  Z  F  2      �  �  �  �  �  W  �  �  1  �  �  f      �   �  ~  l  Z  H  <  2  '      	  �  �  �  �  �  �  �  �  �  �  \  P  D  7  *        �  �  �  �  �  �  �  y  j  k  m  n  �  �  �  �  v  \  <    �  �  u  5  �  �  k    �  %  �  $  s  �  �  	      �  �  ?  �  n  �  J  
�  	�  �  �    e  �  �  �  �  �  �  �  �  �  u  f  V  E  3  "    �  �  �  �  h  �       �  �  �  �  �  �  �  y  {  �    j  L  ,  �  :  �  �  �  �  �  �  �  �  �  �  �  r  ^  I  0    �  �  �  C   �  �  �  w  m  c  Y  M  B  6  *        �  �  �  �  �  �  u  �  �  �  �  �  �  �  �  �  _  0  �  �  �  ^  !  �  :  S   �  a  ^  [  V  P  G  8  (    �  �  �  �  �  c  <     �   �   �  K  �  �  �  �  �  �  �  [  *  �  �  x  S    �  !  �  �   �  �  y  q  h  [  N  A  5  (        �  �  �      8  R  l  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  ]  D  +    �  �  �  �  x  N    �  �  o  (  �  �  u  I  v  r  n  i  a  W  N  B  6  *  -  =  N  Z  _  c  c  I  /    :    �  �  l  ,  �  �  c  .  F  #  �  �  �  D    �  r    a  =    �  �  �  �  f  B      �  �  �  �  �  e  b  X  >