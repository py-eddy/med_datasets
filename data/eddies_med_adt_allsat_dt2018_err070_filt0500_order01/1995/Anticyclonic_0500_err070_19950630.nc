CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ě��S��      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N
�Z   max       P�f      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       =Ƨ�      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F�z�G�     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q��    max       @vyp��
>     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @���          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��j   max       >S��      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B2�B      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B2�R      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?-V�   max       C��F      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?/ޠ   max       C�hs      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N
�Z   max       P&��      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��$�/   max       ?�E8�4֢      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       =�      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=q   max       @F�z�G�     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?θQ�     max       @vyp��
>     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @P�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ͻ        max       @��           �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         @�   max         @�      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*0U2b   max       ?�Dg8}�     �  QX   	            
         �   
   K   =         !      	   G         	      	   )                     !                                             �      '      +   	   +   2   !            BN��SN��O�0�OiϮOM<HO�39O�-�POm�OF�OP�fP�zCN
�ZO���O�0�O_��ONERP��*NH�pOU�VO-p7O:1�N���P`�OO@�N��-P�N݋[N��O5��O���O�+O$�>N�O?.O�G�O�nO��GO��P"VO"�N�iO�V�N�O.l)O���N���O�N�N��O���N�hOt �O�2WN��dOC"{N1��N�7O�UO���u�D���D���49X�t��o:�o;D��;��
;ě�;ě�;�`B;�`B<o<t�<t�<#�
<#�
<D��<D��<T��<e`B<e`B<�o<�o<�o<�9X<���<���<�/<�/<�`B<�h<��=o=\)=#�
=#�
=#�
=#�
=,1=,1=0 �=H�9=P�`=e`B=e`B=m�h=}�=�o=���=��=\=\=Ƨ�=Ƨ���������������������fgot���������tkigff������+/#�����6<HLUXanz�����zaUHA6C@EIWfknpsu}~{vnkbUClin���������ztrprpnlKR[gt���������g[VPNK126BN[g��������xg[;1�����������������������)0-5NWVB5)
����igt}�������������{yi������������������������)/20*%��)AJKRODC=;5)JDFOR[h��������tmhOJ"/15<?FLJH;"�����������������������������������������������������������������������������#%/<HQUQHE</*# ��������������������������)/4:;6���336BU[^baaa^[OGB?=93�����������������������
#/9?DB<#����������������������)5655/)��������������������#05IPRPKD<0#����������������������������

����ZZ[]`cht������toh[ZVOP\hu{���������uh\V;5<BJTgt|���{umbYNB;PV`cbfmz��������maTP��������������������� �����!)5PSSQMJB)��������������������./9<<FHUUUH<5/......iht���������������ti]bnt{����{nb]]]]]]]]��������������������������

�����
##&(('#!
+%'*3;HT_cdaYTKH;7/+������������������������������	�����������������������������#),1685)��������������������

��������&'*.68ABO[bb[OLEB6)&AABO[^[ZQOHBAAAAAAAA�����	���
		
)6BFGGB<0)
�$�0�<�=�@�=�:�0�$�����"�$�$�$�$�$�$ÇÉÓ×ÓÓÑÇÇÇ�z�u�n�g�f�n�zÃÇÇ���)�7�A�=�)���������������������������������������������������������������Żx�����������ûػлû��������x�m�l�c�c�xFF$F'FFE�E�E�E�E�E�E�E�E�E�E�E�E�E�F�����������������������������������������6�O�hĊĔĕđĄ�t�`�9�6�2����������6�	��"�3�7�.�-�+�)�%�"��	� ���������	�I�m�zŐśŔ�{�b�R�D�#�
��ĺĸĿ������I�����������/�E�J�D�/�	�����������v�������uƁƎƚơƚƎƂƁ�~�u�t�u�u�u�u�u�u�u�u��������������������������������������	�"�;�G�W�a�m�m�T�;�"�	�����ھҾھ�	�����������������������������������������	��"�/�1�&�"��	���������������������	�����B�[�tāĔč�h�)������ùçáù���*�6�C�J�O�Q�O�C�6�1�*�&�*�*�*�*�*�*�*�*�(�5�A�E�N�X�Z�_�`�]�Z�N�A�,�(�����(��(�*�5�@�A�K�A�8�5�(������������������������������������������������ɼʼмμʼ��������������������������	�;�G�`�h�o�n�i�a�T�;�"�	��������׾����ܾ׾ʾ�����������������������¿������������������¿²¨²³¾¿¿¿¿�������ʿ̿Ŀ������y�`�;�.����.�`����/�<�H�K�J�H�B�<�8�0�/�(�#��� �#�&�/�/�M�Z�d�f�f�f�d�[�Z�N�M�A�9�6�A�C�M�M�M�M�������ûʻû��������������|�x�q�m�x����������������������������r�f�\�R�X�f�r��4�A�H�C�;�7�6�4�)�(�%����������(�4�4�A�M�Z�f�s����w�f�Z�M�A�4�*�(�'�(�,�4�-�:�?�F�S�_�e�d�_�^�S�Q�F�@�:�1�-�*�,�-������
�	�	�����������߾ݾ۾߾����� �(�5�A�D�A�>�5�(����������ŇŔŠŭŹ������ŴŭŔŇńŃńŊňłŁŇ�`�l�y�����������������������y�`�S�M�S�`��*�6�C�M�V�b�h�h�\�6���
��������Ƴ����������/�0������ƳƧƛƇƁ�xƁƳ��(�4�A�I�M�E�A�4�0�(����������(�,�4�9�4�(�%�������%�(�(�(�(�(�(�"�/�;�E�G�E�;�/�&�"��	��������������"�������������������������������������üʼӼּڼ����ּʼ���������������D{D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{D{�������	���	��������������������������ĳĿ�������
��������������ĿĻĴĳįĳ�zÆÁ�|�z�u�n�a�U�K�U�W�a�j�n�y�z�z�z�z�����������-�+�"�������������¼�����˾���&�(�4�A�M�N�N�M�A�4�(�������S�_�l�x�����������ûϻû��������u�m�^�S�ܹ����!�'�-�'�!�������ҹùĹѹ�EuE�E�E�E�E�E�E�E�E�EuEnEjEkEuEuEuEuEuEu����'�2�4�@�H�K�@�4�'�����������f�r�y�y�v�r�h�f�Y�Q�Y�^�f�f�f�f�f�f�f�f����������������ۺֺɺɺֺߺ�e�r�������ƺ������������~�h�Y�L�D�F�W�e < 7 Z F W P   0 < ? E Y D T [ V U F % : - < " a X � g 0 ? % < > = j W ^ " E a 5 l ' � h ( X D x 0 } W O ! L < P =    �  �  �    �  7  c  �  �  �  �  W  5  N    �  �  s  �  s  �     |  �  �  A  '  �  �  /  _  t  (  �  D  m  ^  T    P  |  L  �  �  �    .  �  �  �  3  V    �  G    ���j:�o;�o;o��o<e`B<�j>%�T<D��=��
=�+<t�<�=�w<�j<�C�=��
<�o<�<��
='�<�1=]/<�h<�`B=,1=�w<�h=P�`=m�h=L��=8Q�=�w=�P=@�=<j=aG�=�o=�\)=e`B=,1=m�h=D��=�o>S��=e`B=\=���=���=�hs=��>1'>%=�F=��`=�h>%�TBX�B
�B7�B}�B'�.B� B	�WB	�B A�B��B.rB �B�&B�B��A��B78B��Be�B�]B�SB"��B�B��B�uB[�B��BOB!��B%�B ��B#��B0�B2�BB��A��wB,�BB|�B�yB�vB�$B
ڬB(fBB]B�BI�A�x�B��B,�B3%B~vBI�BTSBUB��B��B�BLB	ݬB;�B�QB'�NB>�B	?�B	=�B �BA�BBrB �B�mB B�~A���B�ABB�B�{B��B�cB"�`B
�BݚB GB�B�dBkB!�B%�B ��B#��B6�B2�RB��B < B,�,BL�B��B�B�9B
CB(�	BB�B�BGQA��B^^B?�B:^B�B@ BEB��B��B��B@MB	��AȄRA���A�b�@���C��FA�`A�eFA\�rA��A��]B�A��?A^��As�A��AԺnB z�A��A�ǆA���@��A`�_AN\�A���Aj�A���A=d�@�Õ@銀A2�+A=�@��uAW�A���A��5AR�A��B�+A7%�A6:A��A@S�@�ZC��IA�#�A弮A�i'A���A9�@��?-V�C��@�t�@�3X@M�@��B
;A�}�A�x�A�~�@� �C�hsA���A�}7A]&A�A���BD�A�s6Ab�As-�A��NAՀ�B D�A�A�A�5A҂�@���A`D�AN��A�|�Ai�A�A>��@��B@��A2K�A>��@�A�AV�A��A�A�7A���BUA76�A6�A��J@S�@��C��A�رA冴A�|�A���A8��@��V?/ޠC��@���@�%�@O�Q@�   	                      �   
   L   =         !      	   H         	      	   )                     "            	                                 �      (      +   
   +   3   !            B         %               1      ;   7         '         9                  %         /                                 #      -                                                                                    -   #         %         #                           !                                       '                                                      N��N-	OG��OiϮN��^O�39O�O[��N�˜P&��O�;�N
�ZOe��O�9�O#Q�Os�O��NH�pO�;OĵO ��N���O�4O;`QN��-Op�tN��	N��O5��O_�O2!O�sN�O?.O?�RO�nO<�OyE�P#O"�N�iO�V�N�O.l)O
��N���Oi�N�rO��NZ��OP��OW��NҮlOC"{N1��N�7Os��  �  V  �  �  �  %  �  4  �  �  �  �  �  f  8  �  q  ]  �  �  e  �  O  �  �  �  D  �  �  �  �  �  �  q  �  �  �  o  �  <  D  	  �  �    �    �  j  L  �  �  	�  �  T  �  	�����`B��`B�D����`B�t�;��
=� �;ě�<�h<�h;ě�<#�
<o<49X<49X=C�<#�
<e`B<T��<���<u<���<u<�o<���<�1<�9X<���<��=t�<�h<�`B<�h=C�=o='�='�=0 �=#�
=#�
=,1=,1=0 �=�=P�`=�7L=}�=u=�%=�C�=�E�=Ƨ�=\=\=Ƨ�=�S���������������������rnrt�������trrrrrrrr�����	
��������6<HLUXanz�����zaUHA6PNPUVabnovurpnbUPPPPlin���������ztrprpnlYW[[gt��������tg`[YYMHGHLNY[gtx~��|tg[NM�������������������������)>EI5)	���������������������������������������������	),./-)#���)BIJRNCA;95)IJOX[ht�������th[VOI""//29;=B;;/"""��������������������������������������������������������������������������������#/<EHMQLH?<2/($#������������������������%*022)����556BR[]a```[OIB@=:65����������������������
/47;<<</#
���������������������)5655/)��������������������$0<IMLJG@<0*# ���������������������������


�����ZZ[]`cht������toh[ZVOP\hu{���������uh\VA;?BNZ[gt~wrkg_[TNHAPV`cbfmz��������maTP����������� ��������������� �������
 #)5LPOQNJB)
��������������������./9<<FHUUUH<5/......iht���������������ti]bnt{����{nb]]]]]]]]������������������������

 �������
##&(('#!
7224:;GHITY^_[TPH;77�������������������������������������������������������� ���	)/673) �������������������������

���������&'*.68ABO[bb[OLEB6)&AABO[^[ZQOHBAAAAAAAA�����	���)6;ACCA<6)�0�9�=�?�=�9�0�$�����$�&�0�0�0�0�0�0�n�zÇÊÉÇ�}�z�z�n�k�l�n�n�n�n�n�n�n�n����&�'������������������������������������������������������������������Żx�����������������������x�v�t�s�x�x�x�xFF$F'FFE�E�E�E�E�E�E�E�E�E�E�E�E�E�F�����������������������������������������6�B�O�[�h�l�t�u�u�l�h�[�O�B�6�+�#�'�4�6���	��"�.�'�"�"��	�	�	�����������������0�I�U�a�k�n�j�[�F�0�#�
��������������0�������	��!�*�+�"��	�������������������uƁƎƚơƚƎƂƁ�~�u�t�u�u�u�u�u�u�u�u������������� �������������������������ſ	�"�;�G�V�`�k�`�T�;�"�	�����ݾԾܾ�	�����������������������������������������������	��"�,�#�"��	����������������������)�P�S�P�C�)��������������������*�6�C�J�O�Q�O�C�6�1�*�&�*�*�*�*�*�*�*�*��(�5�A�N�Q�Z�Z�]�Z�W�N�A�5�3�(������ �(�5�>�A�G�A�5�3�(����������������������������������������������żʼμ˼ʼ��������������������������.�;�T�Z�`�c�^�T�G�.�"�����������	��.�׾�����پ׾ʾ��������������������Ҿ�¿������������������¿²¨²³¾¿¿¿¿���������������y�m�`�T�;�9�-�1�P�`�t�����/�<�G�G�?�<�3�/�$�#�"�#�$�)�/�/�/�/�/�/�M�Z�d�f�f�f�d�[�Z�N�M�A�9�6�A�C�M�M�M�M�������ûʻû��������������|�x�q�m�x�������������������������r�f�a�W�Y�`�f�r�������(�)�1�*�(�!���������������A�M�Z�f�l�s�|�s�s�f�Z�M�A�4�.�+�4�5�A�A�-�:�?�F�S�_�e�d�_�^�S�Q�F�@�:�1�-�*�,�-������
�	�	�����������߾ݾ۾߾������(�5�5�8�4�(�������������ŇŔŠŭŹ������ŴŭŔŇńŃńŊňłŁŇ�`�l�y�����������������y�l�`�[�T�S�S�Y�`�*�6�@�C�K�T�`�\�S�6���������	���*ƚƳ����������&�*��������ƳƠƌƇƎƚ��(�4�A�I�M�E�A�4�0�(����������(�,�4�9�4�(�%�������%�(�(�(�(�(�(�"�/�;�E�G�E�;�/�&�"��	��������������"�������������������������������������üʼӼּڼ����ּʼ���������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��������	���	������������������������������������� �
��
����������������������n�z�{�z�x�p�n�a�`�Z�a�k�n�n�n�n�n�n�n�n���������
��,�)�!�����������¿½�����˾4�A�M�N�N�M�A�4�(����(�)�4�4�4�4�4�4�_�l�x�����������û��������w�o�l�a�_�X�_�ܹ���������������׹ȹȹϹչ�E�E�E�E�E�E�E�E�E�EuEpElEuEuE�E�E�E�E�E�����'�2�4�@�H�K�@�4�'�����������f�r�y�y�v�r�h�f�Y�Q�Y�^�f�f�f�f�f�f�f�f����������������ۺֺɺɺֺߺ�r�~�������������������~�r�e�X�P�U�Y�e�r ; - X F 2 P    > O > Y B S S W ) F + 7 ) :  a X x \ 0 ? + 0 ? = j N ^  5 ] 5 l ' � h  X , U 0 j P <   L < P 1    �  M  �    �  7  8  �  �    f  W  �  ,  �  D  $  s  U  I    �  �  �  �  s  �  �  �  �    %  (  �  �  m  �  	  �  P  |  L  �  �  (    H  E  �  �  �  �  �  �  G    �  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  @�  �  �  �  �  �  �  �  �  �  �  �  s  e  Q  =    �  p  /   �  +  +  ,  .  1  7  @  G  N  T  O  >  '    �  �  h  �  `  O  �  �  �  �  �  �  �  �  �  �  �  �  �  f  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  Z  6    �  �  X    �  �  �  �  �  �  �  �  �  �  �  �  �  w  r  n  D     �   w  %  �  �  �  g  *  �  �  S    �  �  &  C  4    �  �  I  �  �    J  v  �  �  �  �  �  �  �  �  �  �  r  M    �  E  �  	8  
�  �  }  K  �  u  �    $  4    �  �       �  	�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  m  ;  	  *  �     Q  �  �  �  �  �  �  �  �  a    �  M  �  z  �  �  �    N  t  �  �  �  �  �  �  �  �  r  ;  �  �    �  �  !  �  �  �  �  �  �  �  �  �  �  �  �  �  |  h  S  ?  *      �  �  �  �  �  ~  k  U  ;    �  �  �  �  �  v  9  �  r  G  \  b  Q  9      �  �  �  �  �  �  i  C    �  l    �    7  3  5  7  7  1  (    
  �  �  �  �  p  L  '    �  �  L  �  �  �  �  �  �  �  �  �  �  q  V  ;    �  �  �  �  �  r  �  �  �    2  W  o  o  _  C  !  �  �  X  �  k  �  �  �  m  ]  _  `  a  ^  Z  V  P  J  D  H  V  d  o  s  w  {    �  �  �  �  �  �  �  �  �  �  y  \  6    �  �  Y    �  o  C  8  �  �  �  �  �  �  �  u  _  H  /    �  �  �  �  �  f  <       ?  S  ]  c  c  Y  C  $  �  �  �  5  �  Z  �  K  �  �    �  �  �  �  �  �  �  �  �  �  �  �  w  ^  D  +          �     "  :  H  N  N  H  ?  2  &    �  �  �  T  �  �    �  �  �  �  �  �  z  l  \  J  4    �  �  �  q  D    �  �  P  �  �  �  �  �  y  ]  ;    �  �  U     �  �  �  @  �  �  Y  S  P  c  m  o  w  �  �  �  h  >      �  �  L  �  V  �   �  �    #  8  C  C  =  -    �  �  �  �  o  J  "  �  �  �  �  �  �  �  �  �  �  �  �  u  e  T  C  1      �  �  �      �  �  �  �  �  q  I    �  �  �  �  U    �  �  }  I    �  �  �  �  �  �  �  �  �  z  >  
  �  �  �  �  �  �  P  �  K  �  �  �  �  �  �  �  �  �  �  �  �  �  t  H    �  M  �  D  �  �  �  �  �  �  �  �  �  �  ~  n  \  F  +    �  �  <  �  �  �  �  �  �  y  \  <    �  �  �  v  I    �  �  �  �  �  q  c  V  G  7  '      �  �  �  �  {  Q  &   �   �   �   �   y  �  �  �  �  �  �    e  I  (    �  �  �  z  K  :  M  s  �  �  �  �  �  �  �  �  �  q  P  +    �  �  �  V  #    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  W  6    �  �  �  =  k  W  [  g  [  I  /    �  �  �    S    �  �  ,  �  �  �  �  �  �  |  a  A    �  �  �  �  �  �  L    �  �    �  <  -      �  �  �  �  �  ~  ^  ;    �  �  �  W    �  N  D  =  6  /  )  "            �  �  �  �  �  �  �  �  �  	  �  �  �  �  �  �  �  �  j  J  '    �  �  �  �  W    H  �  �  �  �  �  �  �  y  f  T  C  5  '    #  '  -  ;  I  W  �  �  �  �  q  U  6    �  �  �  �  Z  (  �  }  �  O  �    �  �  D  �  ?  �  �  �      �  {  �  +    �  �  �    �  �  �  �  �  �  �  �  �  }  n  `  Q  @  .    	  �  �  �  i  E  �  �  �        �  �  �  �  k  ,  �  T  �  �  �  �  �  	l  	�  
y  
�  6  �  �    :  n  �  �    9  q  �  �    V  �  b  i  c  W  K  /    �  �  �  h  7  �  �  n    �  �      �     G  :  '    �  �  �  �  �  �  �  z  h  f  �  �  �  �  u  �  �  �  ~  \  1    �  {  3  �  �  i  �  �    �  L  a  �  �  �  �  �  �  r  ?    �  ~  .  �      z  �  O  �   `  	�  	�  	�  	�  	�  	t  	T  	/  	  �  �  6  �      �    T    �  �  �  �  �  _  7  	  �  �  v  9  �  �  a    �  i  C    �  T  D  5  &        �  �  �  �  �  �  �  �  �  s  `  M  9  �  �  �  �  �  w  P  !  �  �  �  =  �  �  E  �  �  3  �  d  	N  	�  	�  	�  	�  	�  	�  	�  	}  	E  	  �  z    �  <  �  �  �  �