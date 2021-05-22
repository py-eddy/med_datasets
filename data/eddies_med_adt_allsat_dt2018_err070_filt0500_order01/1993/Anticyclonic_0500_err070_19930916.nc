CDF       
      obs    A   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�I�^5       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       PqM�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �<j   max       =Ƨ�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @E���R     
(   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vs��Q�     
(  *�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  5   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �0 �   max       >~��       6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��)   max       B,C�       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B,J"       8�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       =��x   max       C��e       9�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       =�   max       C���       :�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       ;�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7       <�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          7       =�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       PXܡ       >�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�֡a��f   max       ?�!�R�<       ?�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �<j   max       =��       @�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E���R     
(  A�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ۅ�Q�    max       @vs33334     
(  K�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @P�           �  V   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @���           V�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >\   max         >\       W�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�C,�zxm   max       ?�!�R�<     P  X�                        &   2         	   
               +   �   
   
               *   -                                                   M                        '   "               \   [   7   ;      2   �   N)h!Ni?dN��+On(�Of@N�
�O?��Pq�O�AVN��KNl��NI�'N�e�NBS,M��pN���N_�sO���PqM�N��1O�N7�CN�U�N�=,O��O��O���O~��O5o(OJ�Ng*Nǎ�O �N�f�N���Nf~�OTN0lzO�`�P��N�N�N���N͖�P�<N�'O�RNYQ3N �N9BN�N��SP#ĚPXܡM��M�,�NC��OD��PGTP]b+O��Pn4N�;�OF�O߿|N7E��<j�e`B�D���o�ě��o��o%   :�o;o;ě�;�`B<t�<#�
<#�
<#�
<D��<D��<e`B<�C�<�t�<���<��
<�1<�1<�j<ě�<ě�<ě�<���<���<���<���<���<�`B<�`B<�h<�h<�=o=o=+=\)=\)=\)=\)=��=��=�w='�='�=49X=@�=T��=Y�=Y�=]/=aG�=aG�=y�#=�o=��-=���=�1=Ƨ���������������������JMNU[[[gpog][NJJJJJJ#/<?HISRJH<;0/-(##0<DOY_ZUI<0+('%XWW[]gt��������tng^XJIU`anpopnaUJJJJJJJJEHKCN[ghtw�����tgQNE


"/<@PQH@/"����������������������������������������WVVXZ[`chltuthf[WWWW')-6;BEB@6)''''''''��������������������//5BNQSNB5//////////WUY[bdhlhhd[WWWWWWWW�����������������������������������������
������><@M[g����������t[J>#09;30#��������������������55;<ILNOI<5555555555��������������������mmlmvz�������ztmmmmm��������������������OW[h�����������thXOO��������
 
���#/<HPU^YTH</#enomtz����������zne��������������������)1/)	))*)%$)58=@>5)$��������������������GDDEHRTWabaa\THHGGGGlltv}�������tllllll����������������������������������������jfqxz������������zmj��#/HW^ZHHa]H</#��ABFO[hjjha[OLHGBAAAA2*5=BFNTSNLB<5222222���!������ )5BTXWOB)	�����������)6BNRSPOB6)��������������������/.07<HC?<0//////////����������������������������������������������������������������5BVVB97)��������6IQO6���������������������������a_]]abenppnaaaaaaaaa;;<EHUYZUHA<;;;;;;;;��������������������eebz������������~vme����#)3:@NIHB6������)5980(&"�������);;6)����������� ������������������

�����������

����#"#..%#########��������������������������������������������������������������������������������������������������
�������������x�l�_�F�?�=�:�@�F�S�_�l�x�}���H�U�a�t�zÀÃ�~�z�n�^�U�R�<�/�-�/�4�<�HE�E�E�E�F FE�E�E�E�E�E�E�E�E�E�E�E�E�E��"�/�;�H�R�L�H�H�?�;�2�/�"���
����"���	�"�/�T�a�m�o�n�a�T�;�/���������������x�������Ϲܹ����ܹ������������x�l�e�xE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;���(�4�=�4�(�!������������ÇÓÜàèà×ÓËÇÁÅÇÇÇÇÇÇÇÇ����!�-�:�C�F�G�F�:�-�!��������`�m�p�r�m�m�`�\�W�V�`�`�`�`�`�`�`�`�`�`�G�T�`�a�`�T�O�G�;�8�;�B�G�G�G�G�G�G�G�G���!�&�-�:�=�C�:�-�"�!��
����������������������������������������������������)�.�5�9�2������������������������)�6�O�a�l�w�|�w�k�O�I�)�������������ּ����������ּҼ˼˼ּּּּּּּ������������
��"�(�"��	�����������������������������غ׺�����������zÇÓàì÷ìãàÓÇ�z�y�z�z�z�z�z�z�z�����������������������s�p�n�s�~��������������������/�@�J�B�/�	�����������������I�U�Y�^�[�^�]�P�C�<�/�#����(�/�<�F�I�����ʾ�������	���㾾���������������ݿ������"���������ܿпҿпֿ�����(�5�<�A�G�N�Q�N�A�5�(�����
��M�Z�s���������������x�s�f�b�Z�X�M�J�M�r�������������}�u�r�r�r�r�r�r�r�r�r�r�u�yƁƎƎƚƦƚƐƎƂƁ�u�n�h�e�h�h�s�u�����������������������������Ƽ���������������������������������������������tāčĚğĦħĦĜĚčċā�{�t�t�t�t�t�t�������ûлڻлλû����������������������������Ŀѿֿѿ˿ƿĿ������������������������������������������ŇŔŭ������������ŹŭŠŔŇń�~Ł�~ŅŇ�T�a���������������m�:�"������/�H�T���ʼ�������������������������������������(�,�4�6�4�+�(����������������������������������������������������N�[�g�t�[�B�)������)�N�.�;�G�G�K�G�;�3�.�"��	���"�"�.�.�.�.�ʾ׾������������׾ʾ����������������a�n�z�y�o�n�a�_�`�X�a�a�a�a�a�a�a�a�a�a�r�����������r�n�h�r�r�r�r�r�r�r�r�r�r�M�Y�f�r�w�r�h�f�b�Y�Q�M�L�H�M�M�M�M�M�M�
���#�/�2�/�#���
�	���������
�
�
�
����'�+�4�9�;�9�4�'����	��� ���s�������������������u�g�A�3�+�+�2�B�V�s�"�H�Q�R�G�8�1����������������������"�/�/�#��#�+�/�1�<�=�<�/�/�/�/�/�/�/�/�/�������ûƻû����������������������������(�4�8�4�4�2�(�����&�(�(�(�(�(�(�(�(�S�Z�`�l�y���������������y�l�a�S�M�I�H�S����������#�%������àÓÁÀÊÓàõ�Ż!�S�l���������Żû����:�!����������!Ħĳ����#�7�A�D�B�<�0�#�
������ĴĨĠĦ�~�������úº������~�o�`�Y�L�@�3�/�6�Y�~ǈǔǟǡǩǡǘǔǈ�{�o�b�b�b�i�o�{ǇǈǈEuE�E�E�E�E�E�E�E�E�E�E�E�E�E�EyEuEqEjEuD�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DnDiDrD{D��a�b�k�n�{ņŇōŇ�{�n�a�a�a�a�a�a�a�a�a ; T f Q & R ? I ? . g > ^ F c f `  1 3  & L C � g F * F R � > \ W L [ ( ` ( U S 4 ` ) W : S W ` h K ) s 6 � J ? K L x 2 ` L * M    Q  �    �  �  �  �  �    �  �  Y  �  s  !  �  �  /  �  �  
  I  �  �  Y  �  �  �  �  �  �  �  �    �  �  7  Y  M    �  �  
  �  �    �  "  S  �  (  �  �    A  ^  �  �    
  �  �  �  �  l�0 żo<o;�`B<���;D��<T��=�P=L��<�o<49X<u<���<T��<D��<u<�C�=aG�>e`B<���<�`B<ě�=��<���=<j=�o=�O�=49X='�=o<�`B<�=+=#�
=+='�=#�
<��=ix�=]/=�P=�P=<j=�S�='�=H�9=L��=#�
=,1=H�9=y�#=��=��T=q��=ix�=ix�=��P>�+>�=�h=��#=�E�>o>~��=�l�B��B�+BɲB&!=B	�B�B	-iA��)BN�B`�BtXB��B Q�B��B�B*%	B��B��B
2B%.�B�SB&<B"l"A�ܜBB��BlXB{yB�pB��B�rB�B�8B߬A�$�B��B'�By�B �5B��B,�B��B��BևBڴB�;B�iB&(�B�NB��B!�:B�"B�B?^Bn�BT�B,C�B ��B�Bp�B�>B��BzB�fB�(B��B�B��B&�VB	�B1tB	G{A���BA B@UB6�B�tB ��B��B@�B*A_B��BT�B
 DB%?�B��B&x"B">�A�z�B$�B��B8�BD�B/B��B�2BàB��BøA�}�B�wBE"B��B ɨB��B?�B�-BI�B�sBϸB�B@�B&0�B�VB�cB!��Ba�BL�B?�BF�BF�B,J"BBKB��B��BE�B��B@B:rB��?B!�A��.A҅n@�+nA���C��eA���A�#=��xC�f;A4��A�r@oIoAi�QAf�@k��At�CA��:A֤WA�A�l�@IB5AʈrA�y�A�<�A�/AR�0A�o�A���AC��@�v�BAeBk!A���A�l1@���Av�A���A���A��|@��A5�A�e�A��*A`��AQ�fA��t@�/N@�<�A�ZZ@ƙDA���A��A�d@���A6�>AR8A���@�r�A�M�@�~B�C�GC��A�*}?LPA��sAҀL@�(�AŃtC���A���A��=�C�g�A4b�A�{!@s��Ai)�Ag@s�"At��A��CAց`A��A�jG@I!�Aʖ�A�!2A�wAA�:AR� A�g�A��-AC��@�(3B	�B?)AЍpA�T�@�>�AuSLA�=eA���A���@�B�A6}A�|�A�Q�A_a(AR5�A�}�@珯@�� A�� @��7A�NA��^A�~�@�%�A5K�At#AЍ@���A�~�@U)B�	C�OC��A�                         &   2         	                  ,   �   
                  *   .                                                   M                  	      '   #               ]   \   8   ;      2   �                           )   %                           #   1                  %   !                                          1            %                        '   7               '   3   )   )         !                           !                                                                                                1            !                           7                  +      !            N)h!Ni?dNW�On(�OKNNH��Nۃ�O���O��8N��KNl��NI�'N�e�NBS,M��pN���N_�sO���O��INe�uO�N7�CN�U�N�=,O�N���O^�XO~��O5o(OJ�Ng*Nǎ�O �N]d�N���N+#�OTN0lzO���P��NZy�N���N͖�OځGN�'Op�
NYQ3N �N9BN�N�D8O��JPXܡM��M�,�NC��OD��O���P(O��O�t N�;�O,F%O���N7E�    d  T    �  Z  l  �  C  h  �  q  �    �  7  �  �    m  �  �  �  �  >  �  8  �  �  h    7  j  �  �  L    �  �  $  �  �  �  	�  �  &  �  �  ^  �  �  1  �  S  7  ~  +  �  
C  �  i  �  
�  n  ��<j�e`B��`B�o��o��o;D��;��
<u;o;ě�;�`B<t�<#�
<#�
<#�
<D��<�1=�"�<���<�t�<���<��
<�1=o=0 �=\)<ě�<ě�<���<���<���<���<�<�`B<�<�h<�h<��=o=+=+=\)=L��=\)=t�=��=��=�w='�=<j=q��=@�=T��=Y�=Y�=]/=���=���=�-=���=��-=���=��=Ƨ���������������������JMNU[[[gpog][NJJJJJJ//<DHNLH<40/////////#0<DOY_ZUI<0+('%ZYY[agt��������ytgaZLLUainnnonaULLLLLLLLMNU[gmt}}tg][ONMMMM		"/8=MNIHB;/	����������������������������������������WVVXZ[`chltuthf[WWWW')-6;BEB@6)''''''''��������������������//5BNQSNB5//////////WUY[bdhlhhd[WWWWWWWW�������������������������������������������

���VSW\gt����������tg[V#/060.#��������������������55;<ILNOI<5555555555��������������������mmlmvz�������ztmmmmm��������������������~���������������~~~~�������

���#/<HPU^YTH</#enomtz����������zne��������������������)1/)	))*)%$)58=@>5)$��������������������GDDEHRTWabaa\THHGGGGrmtx�������wtrrrrrr����������������������������������������gryz�������������zng��#/HW^ZHHa]H</#��NKJO[hiih`[ONNNNNNNN2*5=BFNTSNLB<5222222���!����
)5BLQOKFB5)
����������)6<BLQSPOB6)��������������������/.07<HC?<0//////////���������������������������������������������������������������)-.+$"����������6IQO6���������������������������a_]]abenppnaaaaaaaaa;;<EHUYZUHA<;;;;;;;;��������������������xy{~���������������x����)59EGB@6)���������������)5771)������������ �����������������

�����������

������#"#..%#########�������������������������������������������������������������������������������������������������������������x�l�_�F�?�=�:�@�F�S�_�l�x�}���H�U�a�n�z�~Â�}�z�n�a�U�H�<�2�/�.�6�<�HE�E�E�E�E�F E�E�E�E�E�E�E�E�E�E�E�E�E�E��/�2�;�B�?�;�8�/�"������"�%�/�/�/�/�H�T�a�i�m�j�a�T�;�/��	������������/�H�����ùϹܹ߹��ݹϹù�����������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E;���(�4�=�4�(�!������������ÇÓÜàèà×ÓËÇÁÅÇÇÇÇÇÇÇÇ����!�-�:�C�F�G�F�:�-�!��������`�m�p�r�m�m�`�\�W�V�`�`�`�`�`�`�`�`�`�`�G�T�`�a�`�T�O�G�;�8�;�B�G�G�G�G�G�G�G�G���!�&�-�:�=�C�:�-�"�!��
����������������������������������������������������������'�+�(��������������������6�B�Q�Y�\�[�O�B�6�)������������ּ��������ؼּϼмּּּּּּּ������������
��"�(�"��	�����������������������������غ׺�����������zÇÓàì÷ìãàÓÇ�z�y�z�z�z�z�z�z�z�����������������������s�p�n�s�~���������������	��"�*�.�"���	�����������������<�C�H�O�M�H�C�<�/�#�#�"�#�&�/�4�<�<�<�<���ʾ׾����������׾ʾ��������������ݿ������"���������ܿпҿпֿ�����(�5�<�A�G�N�Q�N�A�5�(�����
��M�Z�s���������������x�s�f�b�Z�X�M�J�M�r�������������}�u�r�r�r�r�r�r�r�r�r�r�u�yƁƎƎƚƦƚƐƎƂƁ�u�n�h�e�h�h�s�u�����������������������������Ƽ���������������������������������������������tāčĚğĦħĦĜĚčċā�{�t�t�t�t�t�t�������ûлԻлɻû����������������������������Ŀѿֿѿ˿ƿĿ������������������������������������������Ŕŭ��������������ŹŭŠŔŇŅ�ł�ŇŔ�T�a���������������m�:�"������/�H�T������������������������������������������(�,�4�6�4�+�(����������������������������������������������������5�N�[�g�t�[�B�5�)�$����(�5�.�;�G�G�K�G�;�3�.�"��	���"�"�.�.�.�.�ʾ׾�����������׾ʾ��������������ž��a�n�z�y�o�n�a�_�`�X�a�a�a�a�a�a�a�a�a�a�r�����������r�n�h�r�r�r�r�r�r�r�r�r�r�M�Y�f�r�w�r�h�f�b�Y�Q�M�L�H�M�M�M�M�M�M�
���#�/�2�/�#���
�	���������
�
�
�
���$�'�4�5�4�4�'��������������������������������g�Z�L�G�H�V�Z�g�s���"�H�Q�R�G�8�1����������������������"�/�/�#��#�+�/�1�<�=�<�/�/�/�/�/�/�/�/�/�������ûƻû����������������������������(�4�8�4�4�2�(�����&�(�(�(�(�(�(�(�(�S�Z�`�l�y���������������y�l�a�S�M�I�H�S�������������	��������ñàÕ×ßì�Ż:�S�l���������������F�:�!��������!�:�������������������������������������̺~�������������������~�r�g�Y�L�F�D�L�X�~ǈǔǟǡǩǡǘǔǈ�{�o�b�b�b�i�o�{ǇǈǈEuE�E�E�E�E�E�E�E�E�E�E�E�E�EzEuEsEoEnEuD�D�D�D�D�D�D�D�D�D�D�D�D�DwDpDsD{D�D�D��a�b�k�n�{ņŇōŇ�{�n�a�a�a�a�a�a�a�a�a ; T @ Q ) Q 6 O ' . g > ^ F c f ` "  -  & L C M C E * F R � > \ & L R ( ` & U . 4 ` % W A S W ` h 8 & s 6 � J ? U B 7 $ ` E * M    Q  �  m  �  �  �  �  &     �  �  Y  �  s  !  �  �  �  �  z  
  I  �  �  ;  �  �  �  �  �  �  �  �  t  �  c  7  Y  3    j  �  
  �  �  �  �  "  S  �  �  (  �    A  ^  �  �  �  "  �  �  }  D  l  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\  >\                      &  -  3  6  0  )  #        d  w  �  �  �  �  �  ~  t  j  `  V  L  A  4  (    �  H  �        �    T  P  E  3    �  �  �  q  6  �  �  O  �  �        �  �  �  �  �  �  �  �  �  ~  N        "  +  6  �  �  �  �  �  �  �  �  �  �  �  ~  b  *  �  �    �  �  2  Z  Z  Z  Z  Z  X  U  S  M  C  9  /  $        �  �  E    .  4  ;  J  i  l  h  `  V  F  3    	  �  �  �  �  #  �    Z  �  �  �  �  �  u  S  )  �  �  �  a  %  �  K  �    �  �  �  �    &  3  @  @  7  %    �  �  A  �  �  $  �  &  L  �  h  \  U  O  C  :  3  -  %    
  �  �  �  _  "  �  �  V    �  �  �  �  �  �  �  y  m  a  U  I  =  2  '         �   �  q  l  g  _  U  I  8  '    �  �  �  �  \  *  �  �  �  Z  %  �  �  �  �  t  c  R  C  5  )  G  �  �  �  �  �  |  q  e  X          
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  o  b  U  G  :  -     7  -  #        �  �  �  �  �  �  �  �  �  �  �  o  F    �  �  �  �  {  x  t  q  b  B  #    �  �  �  t  X  >  $  
  �  �  �  �  �  �  �  �  �  �  _  -  �  �  s  #  �  -  �  �    o  {  r  @  �  s  �    �  �  �    `  �  �     !  	�  �  K  U  `  f  k  l  j  e  ^  T  J  ?  3  !    �  �  �  z  U  �  �  �  �  �  �  �  �  y  `  B    �  �  �  ~  t  o  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  m  c  T  D  3  #  �  �  �  �  z  h  R  6    �  �  �  �  N    �  �  \  %  3  �  �  �  �  �  �  q  `  O  <  )      �  �  �  �  �  k  P  �  �  �  �  �  �  #  0  :  6    �  �  �  v  ?    �  u    �    *  T  �  H  z  �  �  �  �  �  y  F    �  I  �  �  %  �  �  �  %  5  6  /      �  �  �  z  :  �  �    T  X    �  �  }  r  d  S  ?  *    �  �  �  �  h  3  �  �  _    �  �  �  �  x  h  T  :    �  �  �  �  t  M    �  �  k  0  �  h  e  b  _  Z  U  M  B  7  +        �  �  �  �  �  `  8    &  ,  2  9  ?  E  4    �  �  �  �  �  t  ]  G  1      7  3  /  +  &  "        	    �  �  �  �  �  �  �  �  �  j  f  c  ^  W  P  G  <  1  )  !          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  d  M  6    �  �  �  �  w  k  _  S  H  =  2  '        �  �  �  �  �  �  �  k    '  =  L  P  J  ;      �  �  �  �  ~  g  Y  W  �  �  �    	    �  �  �  �  �  �  �  �  �  �  i  O  3    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    �  }  x  l  X  >  !     �  �    R    �  �  /  �  |  �  $       �  �  �  �  �  y  K    �  �  �  �  �  a    �    �  �  �  �  �  �  �  �  �  �  �  �  �  t  _  J  -    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    o  `  Q  B  3  $  �  �  �  �  �  �  p  X  3    �  �  �  �  �  �  Z    �  �  	
  	G  	i  	}  	�  	w  	`  	G  	/  	  �  �  b  �  �  �  L  u  �  k  �  �  �  }  s  i  _  R  C  5  "  
  �  �  �  �  z  V  1      $  %  !        �  �  �  �  �  �  {  Z  5    �  �  I  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  Q  E  8  ,         �  �  �  �  �  �  �  �  q  `  N  <  �  �  �  �  �  �  �  �  �  �  w  a  J  0    �  �  �  l  ,  h  z  �  �  �  �  �  �  y  g  K  +    �  �  �  i  )  �  �  �  �  �  �  �    "  /  .  !    �  �  G  �  �  �      �  �  �  �  o  J  %    �  �  �  x  2  T  #  �  �    ,  �   �  S  A  .      �  �  �  �  w  M  #  �  �  �  p  B     �   �  7  +        �  �  �  �  �  �  �    $  ;  M  ]  n  ~  �  ~  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  +      �  �  �  �  �  �  �  �  �  r  M     �  �  r  5  �  �  W  �    O  t    o  Q    �  |  �  .  
O  	f  x  �  �  e  	  	�  
	  
1  
C  
;  
  	�  	�  	K  �  �  ;  �  8  �      �  =  �  &  =  H  �  �  �  �  �  �  �  �  �  %  �     �  �  �  �  3  Y  d  f  i  `  L  /    �  �  @  �  �    �    r  p   �  �  �  �  ~  R  $  �  �  �  |  W  0    �  �  Q    �  �  u  
�  
�  
�  
�  
�  
�  
�  
G  
  	�  	c  	  �    B  H    �  /   �    Q  g  l  n  ^  P  <    �  �  n  �  �  �  �     �    
l  �  B      �  �  �  S    �  g    �  7  �  ^  �  |    �