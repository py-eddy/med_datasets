CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�dZ�1      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��t�   max       =��T      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @E�
=p��     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v|�\)     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @P@           �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @��          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �T��   max       >q��      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�c�   max       B,¹      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��I   max       B,L�      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�l�   max       C���      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�|   max       C���      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P@l9      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?����-�   max       ?�1&�y      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��C�   max       =��#      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @E�
=p��     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @v|�\)     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P            �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˊ        max       @�          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         B0   max         B0      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��s�g�   max       ?�6z��     �  V�                  
      	      7      	      =      E         F                  T   -                
   �   1      9   7   /         7                                    &            +   (   	      +         =N�k�N{)NTniM���Ok��N���O3�NΟ�N	�P_S�OE��Nڰ1O�5mPwҴN�yPx��N�s	O�eP���N���Ox�]N�eN���O{�P3,'P;��Np�vN!-�N��UO��|M� PSO�?�N�#\PJ0�P�O���Nl�XO"ĺO�N�PfO���O�~�N�^|N}�O���Nn��M�}@N�IVN��O���O�x�O��N��2N�LO���O��N���O �O�EN/|NG��OⱣ��t��e`B�D���D���o���
��o��o��o$�  $�  ;o;ě�<o<o<t�<t�<t�<#�
<#�
<49X<49X<49X<D��<D��<T��<e`B<u<�o<�9X<�9X<�j<ě�<ě�<���<���<�/<�`B<�h=+=C�=C�=C�=\)=�P=�P=��=#�
=49X=49X=<j=<j=<j=D��=L��=P�`=P�`=aG�=ix�=ix�=m�h=m�h=��T_\XZZbnouvusonhb____������������������������������������������������������������" #$)/5BNSW[_e^NB5)"��������������������.5BKN[iqtywtpg[NI@5.����������������������

	������������B[t����t[N) ��_^an����������znjga_��������������������WV]at���������|wthbW��������)4DK/#������������������������������
#/7;:/�������������������������`dhipt����������th`����B[fkkf`B)���_[bnpuz{||{nmhbb____967=BHN[gu}~ztg[NHB9��������������������912<HKU_]UH<99999999RT[amz�������zmhaZTR���6O[][TLB6)���������������������������������������568BHOTSOBA655555555}���������������}}}}�����������������������������������������������
#-00-'
���)%+4Hanz}zqnf_^UH<2)�����������������
/<LVVI<</!	��������6=@=:1-)���QTamz�������zmeb[WTQ#(0<C<830-#�����������������������������
�������������������������ggf_it�����������tng����
#0168850#
���� #(/5<HLLH</'#      �����������������������%/6>6)���������������������������������������� )+,*) )35654/)����������wvx~����������"%&)*,)�����������������������|���������������||||�������������������������(,.,)!������	5BNPUB5)����=78BOY[ahuxwytshc[G=�{������������������")/4/"��������������������}������������������}�l�x�����������������x�l�_�Y�_�b�l�l�l�l�����������������������������������������[�b�g�h�m�g�[�N�F�M�N�U�[�[�[�[�[�[�[�[�������������������������������������;�G�T�`�n�y���y�s�m�`�X�T�G�;�3�/�.�6�;�'�.�3�@�A�D�@�9�3�'� �����%�'�'�'�'�������������������������������������������������������������������}�{�~���������@�J�H�@�4�2�'�$�&�'�4�?�@�@�@�@�@�@�@�@������������������ïà�f�`�~ÆÈÓà����F$F1F:F=F&F&F2F1F$FFE�E�E�E�E�E�FFF$��������*�0�2�*�������������������4�M�����ʾھ۾׾��������{�f�4�(���(�4����"�H�[�e�r�v�q�a�;�"�	���������������ĽͽнҽнĽ������������ĽĽĽĽĽĽĽ����5�A�S�`�m�o�j�_�N�5������ٿĿǿ��������������������������������������Һ3�@�H�L�Y�_�e�g�m�q�e�[�Y�L�@�3�-�'�(�3�ݿ����6�H�M�L�5����ݿÿ��������Ŀݽ����������ݽܽнĽ����Ľнݽ߽�����I�U�b�n�r�{�ŃŁ�{�n�b�U�I�;�6�9�<�C�I��(�-�4�7�4�+�(�$�������
����EPE\EiEnEjEiEbE\EPEDEEEGEPEPEPEPEPEPEPEP���������������������������������������������	�/�;�I�V�[�A�/������������������𾌾����˾վо��������s�e�Z�P�V�[�Z�h���������������������������������������������!�'�)�'�"���������������/�7�<�H�I�H�D�<�3�/�#�#���#�&�/�/�/�/�(�A�N�g���������������s�g�Z�N�A����(��)�+�,�)�#��������������DoD{D�D�D�D�D�D�D�D�D�D�D�D�D{DoD]DTDYDo��������������������ùìèô÷���޾�����	��"�.�;�B�;�3�.�"���������`�y�����}�r�T�.���ʾ����׾�	�"�G�T�`�~�����Ⱥɺ����������~�e�L�A�L�P�R�`�r�~ĦļľĽĺĲĦĚčā�s�^�Z�a�h�tāčĚĦ����������������޼������ܹ����
������ܹйù��������ùϹӹ��n�{ŇŔūŹ��ſűŔŋŇń�{�n�^�X�W�d�n�n�zÇÍÉÇ�~�z�n�c�a�\�U�O�U�a�c�i�n�n�����������������������������������������4�A�Z�q�~������s�f�M�A�4�'�����(�4¦®¦¤àìù������ýùìàÓÒÓ×àààààà��-�F�_�v�x�������x�e�F�-�"���������A�M�S�U�Q�M�A�4�0�4�4�9�A�A�A�A�A�A�A�A�/�<�?�C�<�/�#��#�.�/�/�/�/�/�/�/�/�/�/�u��~�x�u�m�h�\�O�M�K�O�X�\�h�n�u�u�u�u���������������������������������������������y�m�`�T�H�@�?�F�T�`�m�y����������������#�4�9�4�������¦�v�w¦¿������ùý��������ùàÔÓÎÇÆÆÓÔáì÷ù�B�L�L�D�B�<�6�-�)�$��&�)�6�6�A�B�B�B�B���ûлѻڻлλû�������������������������#�0�>�F�F�@�0��
��������������������tāčĚġĦįĲĈ�w�i�h�[�Q�N�O�Q�[�h�tƧƳ����������������ƻƳƧƦƧƧƧƧƧƧ������������������r�f�Y�M�K�M�T�Y�`�r��4�@�J�O�\�Y�O�@�4��
���������)�4�#�&�#�����
�
�
��#�#�#�#�#�#�#�#�#�y�����������y�s�r�q�y�y�y�y�y�y�y�y�y�y�4�@�D�M�r�����o�`�S�@����.�2�3�8�2�4 Y F E  ( A K Z k \ G n {   e 7 G n * o  S Q ^ 3 3 I L + \ t $ * u U $ 1 V G 0 i E 0 Q W u G p M B ( � Y � / T J Y u H M i j    �  �  r  1  �  �  �  6  ?  .  �  1  1  �  G  %  �  X  N  �  �    �  �    -  �  F  �  �  S  �  �  B  �  e  g  �  l  �  �  E  �  �  �     �    �  �  �    �  �  �    �  �  �  J  L  �  ��T���o�o�t�<T��;�o<u;��
:�o=aG�<T��<#�
<���=�C�<49X=��-<�1<ě�=��
<�C�=��<T��<�9X<�o=ě�=ix�<���<��
<�/=Y�<��>q��=��=o=��T=��
=��=+=8Q�=�-='�=L��=��=L��=0 �=�+=8Q�=8Q�=T��=<j=�\)=�1=���=q��=�O�=��=�Q�=�%=��T=���=y�#=�o>bNB'�dB)�GB_AB:@B�RB!KBƼBn�B$>�B��B�B�B��BUhB��B��B�lB�B��B(<B�sB��B?A�!B2\B��BQ�BPQBs�BٺB��BG�BV�B+<B�5B�A�bsB%�<BYB<B�B
?GB$��B�vB ��B��B�B�BB�7B�B��B">BkB!Y�BdVB�]BhTB'�B O#A�c�B,¹B4B'�UB*3@B\�B�;B��B!B�B��B$D5B��B#B_BMBE B��B?kB�B��BP�B(?�B��BĕB1HA���B?B��BH�BK�B��B0tB`�B@#B?�B8lB�VB>�A���B%IpB1�B?>B�SB
EB$�AB��B �^BC2B�5B��BA!B��B�*B��B"@uBH�B!@ B? B�BQBBD1B C^A��IB,L�Bߒ@��@�M0A�yB�Af�[?�i�A��3A��@��GA�$�C���A��AD�&A��]A%�HA���AЃ�?�3eA�VA+�A�F<A4�+C���A�Y�A�AGG�Ar5z?��8A�xWA��=AՌNC��qA���A\�]A`�;@E�Aފ\A"Z>�l�A�AǞtA��MA<��A�$�A̎@y�JA;]@AKB(4A��Aj�iA�u�A��A�s�@��_A�YA�*sB�6@�&@�-A�FAz�@Ҟ�@�K�@�W�A��B��AgW?�)�A��<A�i�@ΜBÀ�C���A�+�A?*MA�T�A%� A�xJAЊ�?�5�A�bDA*�UAA3z�C���A�$9A�v!AJ�?Ap��?�C�AA�?MAՇ%C���A�	A^�A]F�@�#Aޏ�A�?�|A�rA�~�A��>A<��A�y@A�~C@�fDA;�A!BŹA���Ai��A�D�A�rRA�@��zA�f�A�x�B$@⚉@ϑ�A��A@�7                  
      
      8      	      =      E         G                  U   -                
   �   2   	   9   8   /         7                            	         '            ,   (   	      +         =                              7         /   1      1         3                  /   -            #      '   !      3   %                              #                  +            !            !         )                              -         )   )      )         !                     )                           1   !                                                            !                     #N�2�N{)NTniM���O*�Nk�<N��5N�ƠN	�P6�OE��Nڰ1OԒ�P*�^N�yP!�N6��NY�LO̝N���OG6�N�eN���O{�O��tP��Np�vN!-�N��UO�5�M� OY6BN�֫N�#\P@l9O�d#Ov�Nl�XO"ĺO$��N�PfO�DRO��9N-b�N}�Os�Nn��M�}@N�IVN��O���O���N���N��2N�LO���O�I�Nvj�O�$O#�N/|NG��O� �  �  %  �  c    �  �  �     |  �  �  6  �  �  �      s  �  �  _  L  �  H  �  �  W  �  \  g  �    _  �     S    _    �  �  i  "  9  c  ^  �  �  D  �  }  Y  B  ,  �  �     �  V  u    
���C��e`B�D���D����o�D��$�  �D����o;�`B$�  ;o<o<�1<o<�`B<T��<u=0 �<#�
<u<49X<49X<D��=0 �<�t�<e`B<u<�o<���<�9X=��#=T��<ě�<�/=�P=t�<�`B<�h=Y�=C�=\)=�w=��=�P='�=��=#�
=49X=49X=<j=ix�=D��=D��=L��=P�`=]/=e`B=q��=�C�=m�h=m�h=�9Xa]Z[\blnsttqnbaaaaaa������������������������������������������������������������$&()-5BNNSX[VNB5))$$��������������������FBDN[dgmtttsg[NNFFFF����������������������

	�����������5[t����tg[N5)�_^an����������znjga_��������������������W[^_lt���������zth[W������
#+4<;3#������������������������������#*/10,#�������������������������lnst�������tllllllll����5BNSSG5)��_[bnpuz{||{nmhbb____<<BBLN[gqt{|wtpg[NC<��������������������912<HKU_]UH<99999999RT[amz�������zmhaZTR)6BIKH@6)���������������������������������������568BHOTSOBA655555555}���������������}}}}����������������������������������������������

�����;45<?HQUaca^UJH<;;;;�����������������
/<KUWSH</#���������)7565)��a\ZYamz���������zoha#(0<C<830-#�����������������������������������������������������������bmt������������tohhb������
#*0450)#
��+&/<HIJH</++++++++++�������������������� � #(,696) ���������������������������������������� )+,*) )35654/)����������wvx~����������� ! ! ����������������������|���������������||||�������������������������(,.,)!�� �� )5BLMB5) �����������>9:BOW[`hswuthe[OHB>��������������������")/4/"�����������������������������������������l�x�����������������x�l�c�i�l�l�l�l�l�l�����������������������������������������[�b�g�h�m�g�[�N�F�M�N�U�[�[�[�[�[�[�[�[�������������������������������������G�T�`�c�m�x�t�m�l�`�T�G�;�7�3�2�;�<�G�G�'�3�=�?�6�3�'�%�����'�'�'�'�'�'�'�'���������������������������������������������������������������~�}��������������@�J�H�@�4�2�'�$�&�'�4�?�@�@�@�@�@�@�@�@����������������ùà�u�n�wÆËÍÔì����F$F1F:F=F&F&F2F1F$FFE�E�E�E�E�E�FFF$��������*�0�2�*�������������������A�o������ľ׾׾���������f�M�4�(�)�0�A�	�"�H�T�`�j�i�a�T�;�/��	�������������	�ĽͽнҽнĽ������������ĽĽĽĽĽĽĽ����5�=�I�Y�\�V�N�A�5��������������������������������������������������Һ@�L�Y�Z�`�_�Y�U�L�@�;�4�@�@�@�@�@�@�@�@�������$�)�(�$�����׿ѿ̿οտݿ�����������ݽܽнĽ����Ľнݽ߽�����U�b�m�n�{�|ŀ��{�y�n�b�U�N�I�@�;�>�I�U��(�-�4�7�4�+�(�$�������
����EPE\EiEnEjEiEbE\EPEDEEEGEPEPEPEPEPEPEPEP�������������������������������������������	��"�)�4�9�9�/�"�������������������������ǾξѾ;������z�s�_�Z�X�`�b�d�u�������������������������������������������!�'�)�'�"���������������/�7�<�H�I�H�D�<�3�/�#�#���#�&�/�/�/�/�(�5�N�g�s�������������s�g�Z�N�A�,� �"�(��)�+�,�)�#��������������D�D�D�D�D�D�D�D�D�D�D�D�D�D{DxDqDvD{D�D�����������������������������������޾�����	��"�.�;�B�;�3�.�"���������`�y�����|�q�T�.�"�	��;þ׾�	�"�G�T�`�r�~�����������������������~�d�[�X�[�j�rāčĚĦĴķĵĭĦġĚčā�t�h�d�`�h�tā����������������޼������ܹ����
������ܹйù��������ùϹӹ�ŇœŔŠŨūŢŠŔőŇ�{�n�k�g�h�n�{�{Ň�n�zÇÍÉÇ�~�z�n�c�a�\�U�O�U�a�c�i�n�n���������������������������������������׾(�4�A�M�Z�j�x���s�f�M�A�0�(�$����(¤¢àìù������ýùìàÓÒÓ×àààààà�-�F�_�q�x�l�^�S�F�:�-���������-�A�M�S�U�Q�M�A�4�0�4�4�9�A�A�A�A�A�A�A�A�/�<�?�C�<�/�#��#�.�/�/�/�/�/�/�/�/�/�/�u��~�x�u�m�h�\�O�M�K�O�X�\�h�n�u�u�u�u���������������������������������������������y�m�`�T�H�@�?�F�T�`�m�y�����������������
���+�(��
��������¿´¼¿������ìù����������ùìàÓÇÇÇÈÓÖàãì�B�L�L�D�B�<�6�-�)�$��&�)�6�6�A�B�B�B�B���ûлѻڻлλû�������������������������#�0�>�F�F�@�0��
��������������������tāčĞĦĬĭĦěĆ�t�h�[�T�Q�Q�T�]�h�tƳ����������������ƼƳƩƳƳƳƳƳƳƳƳ������������������r�f�Y�V�V�Y�a�l�r�y��4�@�B�M�P�V�S�M�@�4�'����	���'�2�4�#�&�#�����
�
�
��#�#�#�#�#�#�#�#�#�y�����������y�s�r�q�y�y�y�y�y�y�y�y�y�y�@�M�Y�r�}�}�l�f�]�Q�@�4����"�+�6�7�@ U F E   = < Z k ^ G n ~  e 8 : S , o  S Q ^  4 I L + N t  = u U  * V G  i F % R W f G p M B ( X V � / T H W n $ M i k    �  �  r  1  g  �    �  ?  �  �  1  v  �  G  �  S  �  �  �  �    �  �  d  �  �  F  �  �  S  �  �  B    �  �  �  l  X  �  0    Z  �  <  �    �  �  �  F  +  �  �    U  �  t  e  L  �    B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  B0  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  e  L  4  %                 	     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  z  w  Y  2     �   �  c  ]  V  P  I  B  <    �  P  �     �  �  	"  	a  	�  	�  
  
_  �              �  �  �  �  r  =    �  �  �  n  O  .  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  �  �  �  �  �  �  �  �  �  �  �  l  O  1    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  n  c  W  L  I  E        !  !  !  !  !  !  !  "    	  �  �  �  �  �  �  �  �  _  w  {  r  V  /  �  �  �  �  ^  �  �  i  k  D    �  �  |  �  �  �  �    b  D  $    �  �  �  �  �  �  �  �  v  e  U  �  �  �  �  �  �  �  �  �  �  w  m  a  Q  A    �  �  L      0  5  )      �  �  �  �  �  �  l  �  �  �  p  O  *  �    K  p  �  �  �  |  f  S  C  -    �  �    8  �  V  �    �  �  �  �  �  �  �  �  �  �  �  �  �  v  f  W  H  9  *    [  �  �  �  �  �  �  �  �  �  <  �  �  Q  	  �  5  �  �  �  �  �  �  �  �  �      $  $  #         �  �  y  _  C  &      	    �  �  �        �  �  �  �  z  j  _  ]  k  �    J  u  �  �  �  +  P  j  s  a  <    �  P  �  P  �  %    �  �  �  �  �  �  �  o  U  8    �  �  �  �  g  ;     �   �  ]  s  �  �  �  }  l  S  2  	  �  �  ]    �  �     �  .  �  _  V  M  D  <  3  *  !         �   �   �   �   �   �   �   �   �  L  <  ,      �  �  �  �  h  @    �  �  Y  �  �  ]  K  O  �  �  �  �  �  �  �  �  �  �  �  �  {  h  V  D  2  !     �  N  
  x  �  �    =  H  @  &  �  �  y    �    O  L    �  �  �  �  �  �  �  �  �  g  ;    ;  I  2    �  �  )  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  v  Y  <    W  R  N  I  C  =  7  0  &      �  �  �  �  �  W  (   �   �  �  �  �  �  �  �  �  �  �  �  u  ^  D  (  	  �  �  a    �  B  S  [  [  V  I  7    �  �  �  b  '  �  �  Y    �  �  �  g    �  �  �  �    5  X  z  �  �  �  	#  	N  	}  	�  
  
�  �  �    f  �  �  \  	  |  �  �  �  H  �  �  �  D  �  �  	|  L  �  �  9  �  �  �  �  �          �  �  Y  �  �  �  �    _  S  G  <  1  $    �  �  �  �  �  x  W  6     �   �   �   j  �  �  �  �  �  w  C    �  �  X    �  w    r  �  R  �  �  T  �  �  �  �  �  �  �  �  e    �  J  �  i     �    �  +  �  0  H  R  Q  C  '  �  �  �  @  �  �  [  �  v  �  �  �  *                  �  �  �  �  �  �  �  k  L  ,     �  _  O  <  $    �  �  �  V  #  �  �  �  �  �  �  p  N  +    �  e  �  �  �  �  
      �  �  |  2  �  p  �  P  T    �  �  �  �  �  �  �  �  �  �  �  �  q  M  )    �  �  �  �  d  �  �  �  �  �  �  �  �  u  W  8    �  �  �  y  S    �  L  K  [  e  i  d  Z  L  ;  (    �  �  �  �  H  �  �  :  �   �  �  �  �    $  2  ?  S  e  n  i  T  1    �  �  k  4  �  �  9  0  (       	  �  �  �  �  �  �  �  e  ?    �  j  �  w  @  E  Z  V  .    �  �  u  A    �  �  Q  E  �  �    �    ^  Q  E  :  0  '              
                �  �  �  �  �  �  �  �  �  �  �  �  t  g  Y  K  >  1  $    �  �  �  |  f  M  /    �  �  �  �  j  X  E  0      �  �  D  ?  :  6  1  ,  '  "            �  �  �  �  �  �  �  �  �  z  s  X  M  >  *    �  �  �  �  x  L    �  P  �  �  *  I  F  6      x  c  V  e  C    �  j  �  n  �  �  4  �  4  W  O  :    �  �  �  d  #  �  �  >  �  w    �    �  =  B  #    �  �  �  �  �  �  �  k  >    �  �  �  R  �  �  J  ,      �  �  �  �  x  M    �  �  H  �  �  I  �  �  g  7  �  �  �  �  �  �  l  S  2    �  �  j     �    �  �  N  $  �  �  �  �  �  �  w  Z  4  
  �  �  K  �  �  N  �  u  �  G                                 �  �  �  v  G  �  �  �  �  �  �  e  =    �  �  ]    �  �  J    �  =    �  �  9  O  S  V  R  8    �  �  �  [    �  d  �  F  ?  %  u  f  W  I  :  ,      �  �  �  �  �  �  `  6     �   �   �        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  
�  
�  
�  
�  
�  
�  
e  
  	�  	_  �  �    �  4    h    #  2