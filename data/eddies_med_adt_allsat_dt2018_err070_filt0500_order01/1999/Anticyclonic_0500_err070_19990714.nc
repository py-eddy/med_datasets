CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?˥�S���      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�`�   max       P�s�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =�C�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @E�z�G�     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vc�
=p�     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @P�           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @�"`          �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       >�=q      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��V   max       B,��      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B,��      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?׮�   max       C���      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?�>   max       C��>      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�`�   max       P�ِ      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�H���   max       ?�_��Ft      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ���   max       >�-      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>\(��   max       @E�          	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�         max       @vc�
=p�     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʩ        max       @�ܠ          �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�e+��a   max       ?�Z�1'     p  S(                        v   	   F   	            @   C                        I       >                  	      	                     %         *      8   	   0             �   c      %   
      h   N��O�cO�O�NFNiraO=6_N.��P�s�Ne4�P=��N��M�`�N\�O��P(B�P01�O��N��O��"O�vHO� O�yAO1m�P��?OjTO�{N �0N8aNׂ1N��WM�&sNQ��OG^N��sNo3N��XO@�N�rO9�O�O�W:N���Nf�CP�#�O���PO��O�c�N2��O�.N<�P6(O�O�s�OV�Nǜ�N�2�Or��N�c���49X�#�
��`B���
�D��;��
;ě�<o<#�
<#�
<#�
<49X<T��<u<�o<�o<�o<�o<�o<�o<�C�<�C�<��
<�1<�j<�j<�j<ě�<ě�<�/<�h=C�=C�=t�=��=#�
=#�
=0 �=0 �=0 �=49X=8Q�=<j=D��=D��=D��=P�`=P�`=Y�=Y�=]/=aG�=aG�=q��=u=�%=�7L=�C�%)05BLNURNB?5)%%%%%%���������������������	
/;?BAADC@;/"	����������������������������������������lnnrz����������zxqnlw~����������wwwwwwww�������"5Nag_J5�����������������������ZX[ft������������taZ#./063/.'$#   #/030/-#          b``bnv{|~{unjbbbbbbb����������
�����������5BN[jopmNB)��������
<8<D2)
���������#<AJMG@/#
���#%/133/#��������������������|}�~���������������|��������

�����det����������������d������    ��������������
4BW`_QF5)����#/<>HJPQJH</#
����
#<DNTQOH<#
�����
�����������%)3666.)
�
!#---%#!




 #)0<IJKIH><0*#�������������������������������������������()***)����/,*056BCFHEB85//////��������������������cins{������������{nc��)5>BIEB5)�����������������������������������������#/97:<=<3/*#����)BIIGA6/)������������������������/++/9<>ADC<<;83/////kmz�������������ztmk��������������������)5EKLMIB5)��LMQU[agot{����xtg[NL���������		�������TQUY[_hiihb[TTTTTTTT'*%')/;HT]nqoeaTH;/'�����  �������������=8;?N[t�������tg[NB=���������	')!����=96;BO[hv|vtnh[OFB=�z}�������������������������������������_^UU`annsszz|zwnha__�����

�������������������������������������������ŹųŷŹ��ſ�����������ƺ����������źǺ��������������������������H�T�^�a�h�p�r�m�a�T�H�;�/�"����/�=�H�����ʼʼѼʼʼʼ������������������������C�O�P�P�V�\�\�\�O�D�C�<�:�B�C�C�C�C�C�C���������������������������������������˺L�Y�\�c�e�n�e�Y�V�O�L�F�L�L�L�L�L�L�L�L����b�{ŇŕŊ�b�T�I�0�������ķįĻ������� �'�0�'���������������/�H�a�m�q�r�p�T�F�/�"���������������/���������������������������s�k�s���������A�N�V�U�N�A�5�3�5�>�A�A�A�A�A�A�A�A�A�A�l�x���������|�x�l�e�_�X�_�k�l�l�l�l�l�l�������ûʻ˻Ļ��������x�l�j�o�x����������������������������������������������)�5�N�g�t�v�n�j�k�[�R�N������������)�m�z���������������������z�Y�J�D�T�[�c�m�"�/�8�;�@�;�3�/�"������"�"�"�"�"�"�����������������������z�m�^�Y�a�f�m�v���M�Z�f�s�z�s�o�Z�R�M�A�4�(������(�M�4�A�M�Z�f�p����{�s�f�Z�A�2����'�*�4�ʾ׾�����	��!���	�����޾׾������ʾ׾����վʾ����������������������ξ�Ƴ�����#�0�$�����ƧƎ�u�h�K�F�M�\�h�uƳ�������������������������������`�m�y���������������m�`�T�G�B�?�F�G�T�`���������������������������������������������¾ʾ;ʾ��������������������������������������������������'�4�8�@�K�K�C�@�4�*�'�����!�'�'�'�'�f�s�w�|�����s�p�g�f�e�f�f�f�f�f�f�f�f���(�)�5�8�5�(�������������U�a�j�m�m�c�a�U�R�H�<�/�+�.�,�2�:�H�R�U���
��#�$�#���
�����������������������x�������������x�w�x�x�x�x�x�x�x�x�x�x�x�������!�&�,�-�0�-�!�������������M�Z�a�f�g�i�f�_�Z�M�A�4�2�,�/�4�>�A�L�M�4�@�M�W�Y�f�k�f�Y�M�@�:�4�*�4�4�4�4�4�4�������(�/�5�>�@�5�(������������EiEuE�E�E�E�E�E�E�E�E�EvEiEeE\E[E\E_EgEi�~�����������ź������~�a�Y�M�L�R�V�^�n�~�����������ý������������}��������������ǈǔǡǪǡǞǔǈ�{�o�b�`�b�o�{ǅǈǈǈǈ�����������������s�Z�A�%�#�-�?�M�d�s�����/�;�H�T�a�m�w�z�g�\�H�;�/�"������/���������ʿ̿пѿ����y�m�`�U�R�S�Y�`�m���ݿ����������������ݿۿֿӿ׿ݽ`�y�������������������y�l�`�U�O�L�S�c�`��'�4�@�I�@�7�4�'�%����������Ŀ���������
�����������ĿĻĶĴĸĳĿ�/�6�<�?�A�<�/�)�#��#�'�/�/�/�/�/�/�/�/���6�B�V�Y�T�G�6���������������������������ʼּ�������̼����������������������������������������r�f�\�S�Z�f�o����������������������������������ǡǭǲǭǭǭǡǖǔǌǈ�{�y�{�}ǈǔǖǡǡ�����ûлܻ޻ܻܻлû�������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ÇÒÓàìíìëàÜÓÑÇÀ�}�|ÇÇÇÇ ^ / 1 Q k  � , b 0 T U B j 1 0 H 0 J M ( _ P 3 <  4 U 6 % _ J 5 Y O m M V N ' @ d � = 8  ( 3 P ( 6 ( L 0 0 G X @ 0    �    !  �  �  �  �    �  9  �    s    �    �  �  "  ]  s  �  �  4  �  �  B  s  �  �  J  q  �  �  .  3  �  �  �  A  e  �  �  �    o  P  �  ]  q  J    k  '  �  �  �  �  ĽC�;�o;�`B��o�o;��
<#�
=��<�C�=���<�t�<T��<�C�=,1=��w=���=#�
<��
=t�=t�=�P=+<�j=�v�=T��=�{<�`B<�h<�h=\)<�h=�P=P�`=,1='�=aG�=Y�=T��=�\)=�%=��
=aG�=D��=�9X=�7L=���=ix�=ȴ9=}�=�{=u>�=q>�=�1=\=�O�=���>/�=���B�B"4A��VB"�8B�B�B�Bf�B"gYB
��B�PB>B(AgB",B,,B$�By�B�=B��BB#m"BZ�B"��B��B2vB�'B�8BvB$�&B%��B7�B��B��B|�B,��B(�B�iB�{B��B�\BeB"�RB�LB�/B�hBu0B	;|B,�.BCfA���B��B	zB��B4gB7�B\�B��BB!iJB�B"9�A��B"�qB@�B�{B��B��B"A�B
GKB�B:�B(>�B"1�B>�B>�B�B��B�[B��B#��B=B#H{B̝B��B��B�-B<�B$�6B%�.B;B�PB��B?FB,��B(B�B�YB��B�oB��B"�B��BK)B�LB?TB	|�B,��B@2A��BB�{B	J>B�kB@�B?�B��B�vB>GB!A�A�a@�A�u�@���B*�A��T?׮�A�@�-]A�A��`A���@�e@��A��SA��2A��A��xA�}A9w�A<�AT��AOA�BR7A���AkE3AJ(�ANGPA��@�]ACJA��yA��hA�{�@��/@^5A<��@��A��C���@�DA �B��A�_%A��Apw�A�5A~r@ˑ�A�U�ApA��@�N�@���AѢxBY�@��C��FA�oA�{m@ 	rA���@�ZB3�A�r�?�>A�~�@�}A�]nA�u%A���@� �@��$A�qA��jA��A���A�m�A8��A=��AR��AO��B��Aғ'Ak�AJ��AN��A�@�70AB�9A���A�QA��@�r@[�A=�y@�[A��#C��>@��A"�`B�WA�;$A��ApoA�=A�@�kA勀A�loAԙ@�}Q@���Aџ�B5�@�5C���Aʅ�                  	      w   
   F   	            @   D                        J       ?                  	      	                     &         +      9   
   0      !      �   c      %   
      i                           E      -            %   )   -   #         !   !   #      ;                                                   '         5      %      "            )   !                                          -                     !                     %      9                                                   %         5                                             N��N�f�Oo��NFNiraO!��N.��P9�Ne4�O�2.N��M�`�N\�O�h<Oԗ�O���O�m�N��N��3OYbDO��QO���O)P�ِO(0 O~��N �0N8aNׂ1NO3�M�&sNQ��O2.�N��sNo3N��5O*)�N�rO9�O�O�m�N���Nf�CP�#�O���O�O��OB N2��OO��N<�O�!7O���O�s�OV�Nǜ�N�2�N���N�c  $  �  R  I  �  �  ^  b  h  �  �  (  X    [  W  �  �  �  �  .  �  �  �    �  �  _  �  #  >  �  9  �  �  �  �  �  �  #  p  ^  X  s  H  �  �  �  X  �  |  m      7  �  �  )  ����ě��o��`B���
�o;��
=ix�<o='�<#�
<#�
<49X<u<��=#�
<��
<�o<�j<�t�<�t�<�t�<�t�<���<�/=,1<�j<�j<ě�<�`B<�/<�h=\)=C�=t�=#�
='�=#�
=0 �=0 �=8Q�=49X=8Q�=<j=D��=y�#=D��=�C�=P�`=y�#=Y�>�-=���=aG�=q��=u=�%=���=�C�%)05BLNURNB?5)%%%%%%��������������������	/;=A@?BA>;/"	����������������������������������������ztpqtz�����������zzzw~����������wwwwwwww�������5>CB5&������������������������git�������������tmhg#./063/.'$#   #/030/-#          b``bnv{|~{unjbbbbbbb������������
�����)5BN[bgg^NB)�������
#&!
���������#/<DHC4/#
��#%/133/#�����������������������������������������������

�����fgt����������������f�������������������������.<LU[XK5)���#/<FHMOHF</#
#/<BFHHC</#
���
�����������%)3666.)
�
!#---%#!




%%/0<@HC<0%%%%%%%%%%���������������������������������������������')))��/,*056BCFHEB85//////��������������������fjnu{���������{nffff�&)5<@BFBB5)�����������������������������������������#/97:<=<3/*#����BGF@<6-)������������������������/++/9<>ADC<<;83/////kmz�������������ztmk����������������������5?CEEA5)�LMQU[agot{����xtg[NL��������������������TQUY[_hiihb[TTTTTTTT63013;HTWahjfaXTHC;6�����  �������������NJILR[gt�������tg[WN�������������=96;BO[hv|vtnh[OFB=�z}�������������������������������������_^UU`annsszz|zwnha__�����

��������������������������������������������ŹųŷŹ��ſ�����������ƺ����������������������������������������H�T�Y�a�f�n�p�m�a�T�H�;�/�"����/�B�H�����ʼʼѼʼʼʼ������������������������C�O�P�P�V�\�\�\�O�D�C�<�:�B�C�C�C�C�C�C�����������������������������������������L�Y�\�c�e�n�e�Y�V�O�L�F�L�L�L�L�L�L�L�L���
�#�<�W�b�a�I�0�#�
��������������������� �'�0�'���������������/�H�V�X�V�O�H�;�/�"��	�����������"�/���������������������������s�k�s���������A�N�V�U�N�A�5�3�5�>�A�A�A�A�A�A�A�A�A�A�l�x���������|�x�l�e�_�X�_�k�l�l�l�l�l�l���������ûȻʻǻ»��������x�q�m�r�x���������������������������������������������)�5�B�N�[�c�a�^�[�P�B�)��������)�m�z���������������������m�a�T�R�X�a�l�m�"�/�8�;�@�;�3�/�"������"�"�"�"�"�"�������������������z�s�m�p�v�z�����������A�M�Z�f�k�Z�O�J�A�4�(�������(�4�A�4�A�M�Z�f�z��w�f�Z�M�A�7�(���#�(�/�4�ʾ׾���������	�����׾ξʾ������ʾ׾ݾ��׾;ʾ����������������������ʾ�Ƴ��������!��������ƧƎ�u�S�M�R�\�uƳ�������������������������������`�m�y���������������y�m�`�T�N�K�I�N�U�`���������������������������������������������¾ʾ;ʾ��������������������������������������������������'�4�@�B�B�@�4�'� ��'�'�'�'�'�'�'�'�'�'�f�s�w�|�����s�p�g�f�e�f�f�f�f�f�f�f�f���(�)�5�8�5�(�������������<�H�U�a�i�l�l�b�a�U�L�H�;�2�/�.�/�3�<�<���
��#�$�#���
�����������������������x�������������x�w�x�x�x�x�x�x�x�x�x�x�x�������!�$�)�!�������������������Z�_�f�h�f�f�^�Z�M�I�A�4�4�.�1�4�5�A�O�Z�4�@�M�W�Y�f�k�f�Y�M�@�:�4�*�4�4�4�4�4�4�������(�/�5�>�@�5�(������������EiEuE�E�E�E�E�E�E�E�E�EvEiEeE\E[E\E_EgEi�~�����������������~�r�d�Y�Q�P�T�Y�`�q�~�����������ý������������}��������������ǈǔǡǪǡǞǔǈ�{�o�b�`�b�o�{ǅǈǈǈǈ�����������������s�Z�A�%�#�-�?�M�d�s�����/�;�H�T�a�m�w�z�g�\�H�;�/�"������/�y�����������������������y�m�^�Z�[�`�m�y�ݿ����������������ݿۿֿӿ׿ݽ��������������������y�l�`�[�X�[�`�l�y����'�4�@�I�@�7�4�'�%���������������������
��������������ĿĽĽĿ�����/�6�<�?�A�<�/�)�#��#�'�/�/�/�/�/�/�/�/����)�6�=�A�>�6�)�������������������ʼּ�����ּʼ������������������������������������������r�f�\�S�Z�f�o����������������������������������ǡǭǲǭǭǭǡǖǔǌǈ�{�y�{�}ǈǔǖǡǡ�����ûлܻ޻ܻܻлû�������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�ÇÒÓàìíìëàÜÓÑÇÀ�}�|ÇÇÇÇ ^ 6 / Q k  � + b " T U B j 2  D 0 L ? # c \ . .  4 U 6 3 _ J - Y O f K V N ' A d � = 8  ( , P + 6  > 0 0 G X ( 0    �  �  �  �  �  V  �  #  �  8  �    s  �  �    Q  �  J  �  5  b  k  �  t  �  B  s  �  k  J  q  w  �  .    �  �  �  A  1  �  �  �    �  P  �  ]  �  J    s  '  �  �  �  �  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  $  !          )  8  G  V  [  X  T  P  L  W  h  y  �  �  k  �  �  �  �  �  �  �  �  �  �  �  |  Z  7    �  �  �  �  @  J  P  O  H  4    �  �  �  }  H    �  ~  .  �  r  �  U  I  F  C  ?  <  9  5  0  +  %                      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  K  +     �  �  �  �  �  �  �  �  �  �  �  �  u  c  O  ;  $     �   �   �  ^  T  I  ?  4  '      
    "  -  &    �  �  �  �  �  �    h  �  �    )  A  T  a  \  C    �  F  �  L  �  B  "  �  h  b  [  S  K  C  <  E  W  H     �  �  �  �  Y  .    �  �  X  �  �    7  \  �  �  �  �  �  \    �  h  �  2  h  s  �  �  �  �  �  �  �  �  �  �  �  �  f  H  )  
  �  �  �  m  D  (  )  )  *  +  +  ,  0  5  ;  @  F  K  Q  V  [  `  e  j  o  X  S  N  I  B  :  2  '      �  �  �  �  �  �  �  �  {  p                �  �  �  �  �  �  �  g  4    �  �  �  �  �    ;  R  [  P  9    �  �  �  j    �  o  �  D  b  �  3  �    ?  O  R  U  V  J  6    �  �  :  �  O  �  �    �  �  �  �  �  �  �  �  �  �  o  S  4      �  �  Z  �  -   |  �  �  �  �  �  �  �  �  �  ~  u  q  n  j  f  f  g  h  h  i  a  x  �  �  �  �  �  �  �  �  �  �  �  _  $  �  �  7  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  -  �  f   �    ,  .  ,  '         �  �  �  �  �  �  z  L    �  {  "  �  �  �    p  c  e  o  v  u  n  m  u  i  Z  W  O  -  �  �  �  �  �  �  �  �  �  �  �  �  �  m  Y  F  5  #         �  �  �  �  �  �  x  F    �  �  C    �  �  �  c  !  �  �  ,  �               �  �  �  k  2  �  �    �  +  �      �  )  c  �  �  �  �  �  �  �  T    �  u    w  �    O  �  �  �  x  p  g  [  N  A  5  (        �  �  �  �  �  �  �  _  Z  U  P  H  ?  7  )      �  �  �  �  ~  _  ?     �   �  �  �  �  �  �  �  �  �  �  �  �    x  s  m  h  R  8      �  �        !  #  #      	  �  �  �  �  �  �  �  Z    >  B  E  H  L  O  R  V  Y  \  ]  [  Z  X  V  T  R  Q  O  M  �  �  �  �  �  �  �  �  �  q  X  =  #    �  �  �  m  /   �    6  2  '      �  �  �  �  �  r  T  3    �  �  �  V  Q  �  �  �  �  �  �  �  �  y  f  R  >  )    �  �  �  ~  O  !  �  �  �  �  �  �  �  }  t  j  _  T  C  /       �   �   �   �  b  [    �  q  a  O  7    �  �  �  Z  *  �  �  �  �  "  p  �  �  �  �  �  �  �  o  V  ?  *      �  �  �  �  �  q  ^  �  �  �  �  �  �  �  �  �  �  }  h  h  \  1  �  �  �  D    �  �  �  �  �  �  k  I  %  �  �  �  o    �  �  5  �  �  z  #    �  �  �  \  .  �  �  }  8  �  �  ^    �  �  k  %  �  n  p  o  h  Q  :    �  �  �  ^    �  s  !  �  o  �      ^  L  :  (    �  �  �  �  �  j  G       �  �  �  c  6  
  X  U  Q  N  J  G  C  =  5  .  &        �  �  �  �  }  a  s  S  2  
  �  �  w  =    �  �  m  *  �  ~    �    p   �  H  9  "  	  �  �  �  �  f  9  
  �  �  z  D  	  �  h    �  �  B  l  �  �  �  �  e  ?    �  �  Z    �    �  �  �  v  �  }  t  j  a  U  I  <  .      �  �  �  �  �  �  �  �  �  �    4  d  �  �  �  �  �  �  �  �  �  e  .  �  �     ]  �  X  V  R  F  .    �  �  �  r  B    �  �  ]  !  �  �  d  #  ;  c  �  �  �  �  �  �  �  �  {  a  >    �  z    y  �  �  |  z  x  u  n  f  j  t    |  w  q  h  _  S  B  2    �  �  �  `    �  D  �  �  <  f  g  I  �  W  �  b    G  �  
v  �  �  �  �           �  �  �  C  �  }  
�  
5  	S  Y  #  �  �    �  �  �  �  �  i  G    �  �  �  O    �  k    �  r  �  7  .  !    �  �  �  �  {  D    �  }  !  �  .  �    �  �  �  k  A    �  �  �  t  ^  B  #    �  �  �  f  ;    �  �  �  s  e  X  >  &    �  �  �  �  �  �  �  �  p  F    �  �    �  �  "  o  �  !  &    �  p  �    L  y  �  �  s  
w  �  �  i  N  /    �  �  �  u  N  $  �  �  �  d  +    �  �  �