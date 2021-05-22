CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?���Q�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N6�   max       P�L�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �t�   max       =�S�      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @D�\(�     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33333    max       @ve\(�     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1�        max       @P�           p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @��           �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �+   max       >��      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�g8   max       B&U       �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B&8g      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?C�   max       C�U      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?O��   max       C�      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          ;      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          1      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N6�   max       P<�      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�s�g��   max       ?㴢3��      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       >!��      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�        max       @D���R     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33333    max       @ve�����     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�        max       @�)�          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         EP   max         EP      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�qu�!�S   max       ?��|���     �  Pl                        ?               =      Y            E         	         -   E   !      (         /      #   	         !         e             	         0            %            �N��NAa�N.�qO8�N���N�1N-��PaҝO�Z�N��O��N��P�
lO@�hP�L�O� �Ou9
N�bP1��O��]N���N��N��O?��O��P?E�O���NE��O庾NjokN6�O�XN��\ONF�N�;.NȨ�O���OCmN�z�Pd�O�w�N���NV�0N^<�N(�N���O>�Nw"O\B�N(�eOg��N	X7N`y�N�?O�@3�t��ě���9X��o�D����`B��o;D��;��
<t�<#�
<49X<49X<�C�<�C�<�C�<�C�<���<��
<��
<�1<�9X<�/<�/<�h<�=C�=�P=�w=�w=�w=#�
=#�
='�='�='�=@�=D��=H�9=H�9=q��=u=u=u=y�#=�t�=��-=��T=�9X=�9X=�j=�j=�v�=���=��=�S�S[gtuvtge[SSSSSSSSSSst{�������ztssssssss,*.0<?EG<0,,,,,,,,,,���������	������KNW[egjttvytkgb[QNKK����������������������������������
#/9CcgfR</
�����/;HMPG712/"	����

 ������������������������������������������������#)6BKQN5����������������)BKVXRJ5)�������������������
)6<FHHDB6)������������������������**%��������������#/890+%
�����4-06:BEOTVPOB6444444� 
#$&#
��
#)010.#
 ���#!"#(-0<IU_`[SIC<0)#����#*/47875/#
 �� 6MUcfc]OB)��)5MS[b[X@5)
����������������������)5BN[t���zt[N5)!��������������������QUanuqnaXUQQQQQQQQQQ������������������-*-/2<HMSQJH</------��������������������������������������������������������b_chmsty�����~thbbbb
#046534)#
=BOS[hkt�����}th[OE=��������������������������)01*0����������� ',+%����(#!)5BFECB654)((((((��������������������xqnquz~����zxxxxxxxx���������������������������	������������������

����#(%#
###########�������()%���##+%###########\UW[acmz��������zma\!"-/051/"')+57BEJIB?5.)''''''����������������������������

�����������������������������������������������z�ÇÌÍÇ�z�v�n�h�n�w�z�z�z�z�z�z�z�z���������������������������������������������������ûŻû»��������x�n�o�x�|��������!�*�+�*� ���������������������������������������������������������������	������������������������������	�)�G�m�����������m�;�.���ݾؾ޾��;�H�T�a�s�������z�a�T�H�/�"����"�0�;������)�0�)���������������������(�5�F�R�V�P�N�5���������������������z�r�m�a�^�T�P�T�T�a�m�z�����������#�<�{Šŭ������ŭń�b�I�<�#�
���������#����*�2�6�>�@�6�*�)������������ �ƚƳ��������������ƧƎ�k�`�O�D�Rƚ�/�;�H�U�a�f�f�d�_�T�;�/�"������#�/�����	��"�$�$�%���	���������������������'�0�4�@�@�@�4�'�����
�����H�Z�W�a�c�_�U�<�/�#�
���������!�/�9�F�H�T�a�m�z�������������z�m�T�7�1�1�6�;�H�T�z�������������������z�y�w�x�z�z�z�z�z�z�(�4�A�M�Z�\�Z�Z�M�F�A�4�,�(�$�&�(�(�(�(��������������ۼּѼԼּ������������������������������������|�r�j�f�r���5�A�V�g�u�����x�Z�A�5����������ʾ���;�R�K�.���	��ʾ��������������ʾs�x�{����������s�_�Z�R�M�A�=�>�C�M�Z�s¿��������¿²ª²³¿¿¿¿¿¿¿¿¿¿�m�y���������������y�m�T�F�<�:�@�G�T�`�m����������������������������������������ÇÎÊËÇ�z�y�w�z�ÇÇÇÇÇÇÇÇÇÇ�������)�1�)����	������������������������� ����������������������������čĚĦīİıĭĦĚĎčā�t�p�r�o�u�xāč�tāăčĒđčāā�t�r�s�t�t�t�t�t�t�t�t�M�Z�[�T�M�A�4�(�����������(�4�A�M�!�-�:�>�F�S�T�S�P�F�:�-�#�!���!�!�!�!�����ɺֺ�����������ɺ��������������#�'�-�/�3�2�/�'�&�#��
������������
�#àìøöìãàÓÇÆÇÇÓÚàààààà���л��������黷���x�_�!����:�_�����(�5�A�K�N�R�Q�V�U�N�A�(�����������(����$�$�$����������������������������������������~�����������������ּ�����������ּּּּּּּּּ��	�	����
������	�	�	�	�	�	�	�	�	�	����������������������������������������EiEuE�E�E�E�E�E�E�E�E�E�E�E�E�E}EoEiEcEiǭǡǔǒǔǘǡǭǲǭǭǭǭǭǭǭǭǭǭǭ�B�N�[�g�q�t�v�t�g�[�Y�N�H�B�<�B�a�X�b�n�{�~ń�{�n�a�a�a�a�a�a�a�a�a�a�aŠŭŹ����������������ŹŭŢŠŜŗřŜŠ����� �
��
�����������������������������"�.�/�0�/�%�"��	��	����"�"�"�"�"�"�{ǈǔǡǭǭǰǭǭǡǔǈǇ�{�y�t�{�{�{�{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DD�D�D�D� f E ( 9 & g : C H _ + 1 i $ @ > ) W L L # ) a 9 ? S C T 6 U h $ * Q a � : G > W n 8 2 f _ d @ / I W Y * S S / /  b  d  @  �  �  t  S  �    �    �  �  �  �  �  �  �  V  �  �  �  �  �  �  �  �  P    �  W  �  �  #  �  ~  �    �  �  �  C  �  �  �  Y    �  4  �  9  �  4  �    �+���㼓t�;D���ě���o;ě�=��<�`B<T��=��<ě�=�t�<�`B=��=49X<��<���=�E�=L��<�/<�=o=0 �=��=ȴ9=�+=H�9=��w=0 �=0 �=� �=��=���=L��=Y�=]/=��T=�\)=�+>#�
=�^5=�+=�+=�\)=��w=ě�>�=��=�`B=���>�=ȴ9=��>   >��B	�B
aB&�B#�B	fB+�B =�B5A�g8B`B�B
�B!dB��B[\B��B'6B!aB��Bq�B8�B$��B$��B&U BgWB�B�WB�BK�BGBC�B^BבB�CB�bB}B<�B$�9B�PB"TcBpcB��B>�B"��BD�B&cB��B��B\BBDhBy?A�u�A�cBZ�B�B�B		�B
A}B%�fB#
�B	;�B��B DsB�|A���BͻB��B>WB�GB7B?\B�+BBXB!��BABD	B�|B$�IB%3tB&8gBA�B��B�dB �B�=BX?B��B-CB��B��B��B�BI3B$�~BEB"?�B@.B��B�B"��B?�B�*B��B�B�B��B��B =�A�zB>�B:�B��A�PAȺm@�r@�;mA�b�A�e�?C�Ac# A�RmA���A�&VA�r}A��A��B��A��qA���@���A�P}A�.�A�ˬA:��Am�@���A�VAR��AA�A���Ajx-AsŘA�j�A�BA���A�� A�z�A7Â@x�$@3P�A�R@A�=U@���A���Aԏ�A!eSA�A��A��kC�UB˵A�)bA�^A�^�A�'XA��QB4}C���A�~�A�s�@�X@���A�]�A�sx?O��AbNA��A���A�}�A�KA�XqA��B�:A�z�A�N�@�.�A�|�A�u7A���A:�A�q@�éA�ήAR(WA@�kA�m�Ai_�At��A�j�AҦqA���A�v�A݅A:W@|�@<c�A�BA�w�@��A�AsA�ΧA"]ZA�GA��tA���C�B��A���A�n�A��XA�t>A�w<B@'C��                        ?               >      Z            F         	         -   E   !      (         /      #   
         "         f             
         1            &            �                        5   !            ;      5            1   !               !   3   !      %         !                           5   #                     
      
                                       /               1      %            '                                                                  %   !                     
      
               N��NAa�N.�qN���N���N�1N-��P&ԾO�HNa-�O~�N��P<�O(E�P��OT��O^�{N�bP��OaW@N���N��N��O�MO�?jO���O�NE��OҲ0NjokN6�OlJN��MO?=�NF�N��VNȨ�O���O
�Ny��O�UxO޺]N���NV�0N^<�N(�N��gO#JDNw"O\B�N(�eOg��N	X7N`y�N�?O�$  �  �  �  :  {  �  �  3  �  �    b    %  b  4  �  u      ;  �  �  O  �  �  �      �    �  U  �  �  �    [  �  L  e  3  �  �  �  �  �    �  G  �  �  �  �  �  �t��ě���9X�o�D����`B��o<D��<�o<#�
<49X<49X<�9X<�t�=L��<���<�t�<���<�<�/<�1<�9X<�/<�h<�=P�`='�=�P='�=�w=�w=]/=<j=<j='�=0 �=@�=D��=Y�=L��=�E�=}�=u=u=y�#=�t�=��w=�{=�9X=�9X=�j=�j=�v�=���=��>!��S[gtuvtge[SSSSSSSSSSst{�������ztssssssss,*.0<?EG<0,,,,,,,,,,��������������������KNW[egjttvytkgb[QNKK�����������������������������
#/>Xba[J</	"/;<=;/'$"	 

���������������������������������������������!)2GHE@5���������
�������)5;DIIB;)����������������������)6:EGGCB6)'��������������������������&&!������������
"# 
�����4-06:BEOTVPOB6444444� 
#$&#
��
#)010.#
 ���#$+00<IUW]YQIH@<0,%#�����#/47874/#
 �6BJQWYZWOB6)��)5FMOSNFB5)���������������������" )5N[gt���xq[N5)"��������������������QUanuqnaXUQQQQQQQQQQ��������������������-//6<HIPMHC<3/------��������������������������������������������������������b_chmsty�����~thbbbb
#046534)#
JFEO[hst���wth[VOJJ��������������������������#%$����������%*("�����(#!)5BFECB654)((((((��������������������xqnquz~����zxxxxxxxx���������������������������
��������������� 

�������#(%#
###########�������()%���##+%###########\UW[acmz��������zma\!"-/051/"')+57BEJIB?5.)''''''�����������������������������

	����������������������������������������������z�ÇÌÍÇ�z�v�n�h�n�w�z�z�z�z�z�z�z�z������������������������������������������������������������������������������������!�*�+�*� ���������������������������������������������������������������	���������������������������"�.�G�m�y���������m�G�.��	�����	�"�;�H�T�^�a�j�o�p�n�m�a�_�T�H�;�5�/�-�7�;�������!����������������������������(�5�E�R�U�N�A�5���������� ��	��������z�r�m�a�^�T�P�T�T�a�m�z�����������#�<�I�b�}šŨŜŇ�n�U�0�#�
����������#����*�0�6�:�6�*�&���������������Ƴ�������������������ƧƚƃƀƁƈƚƳ�/�;�H�T�X�]�]�[�U�H�;�/�(�$�!� �"�$�-�/�����	��!�#���	�������������������������'�0�4�@�@�@�4�'�����
�����<�H�R�O�T�[�^�[�P�<�/�#������������/�<�T�a�m�z�z�~����z�m�a�T�A�;�8�7�;�<�K�T�z�������������������z�y�w�x�z�z�z�z�z�z�(�4�A�M�Z�\�Z�Z�M�F�A�4�,�(�$�&�(�(�(�(��������������ۼּѼԼּ�����������������������������������r�m�k�r�����(�5�A�U�g�s��q�Z�A�5���� �������ʾ�	��������׾ʾ����������������f�s�u���������s�f�Z�S�M�K�D�D�H�O�Z�f¿��������¿²ª²³¿¿¿¿¿¿¿¿¿¿�T�`�m�y���������������y�q�T�G�?�;�B�I�T����������������������������������������ÇÎÊËÇ�z�y�w�z�ÇÇÇÇÇÇÇÇÇÇ������!�%���
���������������������������������������������������������čĚĦĩĮįĪĦĠĚčā�y�t�s�y�{āĂč�tāăčĒđčāā�t�r�s�t�t�t�t�t�t�t�t�4�A�M�Z�S�M�A�4�(��������(�*�4�4�!�-�:�>�F�S�T�S�P�F�:�-�#�!���!�!�!�!�����ɺֺ�����������ɺ��������������
��#�(�/�-�#�"����
������������
�
àìôõìâàÓÇÆÇÇÓÛàààààà���лܻ��������û��������x�e�a�s�������(�5�A�N�P�O�T�S�N�A�(������������(����$�$�$����������������������������������������~�����������������ּ�����������ּּּּּּּּּ��	�	����
������	�	�	�	�	�	�	�	�	�	����������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�EExEuEqElEuEwE�ǭǡǔǒǔǘǡǭǲǭǭǭǭǭǭǭǭǭǭǭ�B�N�[�g�q�t�v�t�g�[�Y�N�H�B�<�B�a�X�b�n�{�~ń�{�n�a�a�a�a�a�a�a�a�a�a�aŠŭŹ����������������ŹŭŢŠŜŗřŜŠ����� �
��
�����������������������������"�.�/�0�/�%�"��	��	����"�"�"�"�"�"�{ǈǔǡǭǭǰǭǭǡǔǈǇ�{�y�t�{�{�{�{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� f E ( " & g : O U K * 1 V   . < ' W 9  # ) a 8 8 - 6 T 1 U h  & ? a � : G 9 Q I 0 2 f _ d > ) I W Y * S S /   b  d  @  �  �  t  S  $  o  �  �  �  ~  b  l  �  �  �  o  �  �  �  �  `  �  �     P  �  �  W  �  �  �  �    �    @  �  B  �  �  �  �  Y  	  c  4  �  9  �  4  �    5  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  EP  �  �  {  r  i  `  W  N  D  ;  1  (          �  �  �  �  �  �  �  �  �  �  w  g  U  @  +    �  �  �  �  |  S  +    �  �  �  �  �  �  �  �  �  �  �  �  e  I  .     �   �   �   �  �  �  �  �  �  �    '  :  7  #    �  �  �  �  q  W  A  J  {  v  r  m  f  ^  V  K  ?  2  $      �  �  �  �  �  f  9  �  �  �  �  �  �  �  �    w  o  g  _  W  O  G  ?  7  .  &  �  �  �  �  �  �  �  �  �  o  V  <    �  �  �  y  L     �  �    .  2  ,  "    �  �  �  �  �  �  �  |  ;  �  :  a  �    R  �  �  �  �  �  �  �  �  �  �  �  �  T    �  Y  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  c  J  1     �          �  �  �  �  �  l  M  .    �  �  :  �  �    �  b  V  L  D  7  )      �  �  �  �  �  �  }  c  N  9  #        �  	    �  �  �  c    �  #       �  �  D  �  D  �  "  $  #           �  �  �  �  �  �  �  �  _  ?  "      �  /  I  P  S  W  a  b  _  Y  J  %  �  �    �  /  �  {  _  �  �    &  /  3  3  /  %       �  �  �  ^     �  w    j  �  �  �  �  �  �  �  �  �  �  ~  ^  :    �  �  s  8      �  u  m  e  ]  T  K  B  7  -  "        �  �  �  �  5  �  �  �  6  p    k  O  &  )  '  I  `  F    �  q    Y  h  �  J  n  �  �        �  �  �  �  �  c  5  �  �  ^  �  c  �   �  ;  :  9  7  1  )  "    �  �  �  �  �  w  j  ]  T  `  l  x  �  �  �  �  �  �  �  �  x  g  N  0    �  �  �  }  U  *   �  �  u  j  _  S  D  4  %    �  �  �  �  �  �  �  z  f  R  ?  B  J  N  M  H  >  /    �  �  �  �  z  j  �  �  �  �  r  ;  z  �    q  ]  E  '  �  �  �  Z    �  a  �  �    �  �  �  ?  N  q  �  �  �  �  �  �  �  �  ~  J    �  =  �  �  %  :  �  �  �  �  �  �  �  �  �  b  4  �  �  a    �  F  �  {      �  �  �  �  �  �  �  q  Z  B  *    �  �  �  �  �  V  *  �    �  �  �  �  �  �  �  |  ^  :    �  �  S     �    b  �  �  �  �  �  �  �  �  �  u  e  U  E  6  &  �  �  �  T                       $  ,  7  C  N  Y  Z  X  V  S  Q  �    L  r  �  �  �  �  �  W    �  �  W  �  x  �    `  }  8  D  P  S  U  R  G  0  
  �  �  S    �  z  5  �  �  R  �  D  b    �  �  w  ^  =    �  �  d    �  x  .  �  v  E  u  �  �  �  �  �  �  �  �  T    �  �  Q    �  �  v  ?    �  �  �  �  �  �  y  T  /  
  �  �  �  �  �  �  p  K  :  9  D              �  �  �  �  �  �  �  �  �  �  �  b  �  a  [  T  L  @  3    �  �  �  l  (  �  �  *  �  *  �  a  �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  9  �  {  '  %  �  �  H  L  9    �  �  �  u  A    �  �  _  !  �  �  %  �    �  
�  
�  
  
\  
�  V  b  P    
�  
z  
   	�  	&  t  �  �  �  w  P  0  3  1  *      �  �  �  �  �  o  E    �  �  S  �  {  �  �  �  �  �  �  �  }  i  T  ?  )    �  �  �  �  �  �  v  _  �  �  �  o  a  R  D  5  '    	  �  �  �  �  �  �  �  �  �  �  �  �  �  }  ^  ?    �  �  �  �  b  9    �  �  �  �  �  �  �  �  �  �  �  y  h  X  H  ;  0  %  �  �  !  �    A    �  �  �  �  �  �  e  C     �  �  �  m  (  �  e  �  �  !  �  
�  
      
�  
�  
~  
1  	�  	�  	%  �  ]  �  P  �  |  #  �    �  �  �  �  �  s  \  B  '    �  �  �  �  �  m  O    �  �  G  '    �  �  �  j  >    �  �  r  <    �  �  f    �  `  �  �  l  W  H  8  '      �  �  �  �  �  Q  �    �  `    �  �  �  S  :  &    �  �  �  F    �  I  �  _  �  u    �  �  �  u  [  B  +    �  �  �  �  �  �  �  �  �  t  V  8    �  �  �  �  �  �  �  y  f  U  D  2      �  �  �  �  �  �  �  c    �  �  d    �  �  F  �  �  W  �  �  *  �  �    n  �  �  ,  q  �  �      �  �  G  �  �    �  9  �  �  �  �