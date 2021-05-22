CDF       
      obs    I   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�l�C��     $  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       Pz#     $  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��`B   max       =�-     $  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�z�G�   max       @E�=p��
     h  !   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��
=q    max       @vN�Q�     h  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q�           �  7�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @��         $  8|   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �u   max       >�7L     $  9�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�[   max       B,�S     $  :�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��!   max       B,L�     $  ;�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?���   max       C�V�     $  =   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C�^3     $  >0   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �     $  ?T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          9     $  @x   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -     $  A�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P�     $  B�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�n��O�<   max       ?�n��P     $  C�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��`B   max       >��     $  E   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?��R   max       @E�=p��
     h  F,   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��G�{    max       @vN�Q�     h  Q�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q�           �  \�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ʐ        max       @�<@         $  ]�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?:   max         ?:     $  ^�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���n/   max       ?�o hی     �  _�            	   
      	         	            �         	   
      Y                           >   /   *         1   w                     H      	   !                      
      z               	               	            P      C   
   	   �ObFTO�uNk
NeN�~�N%��N��M���N�N[�N�h�O���P'��P&��O��N�_�N��N���N"��PV�N�r5O�QcNS&/N=�NN��]O�7Oa��N3�O�SO�O�O�F�O�1O;�O��Pz#N!�O3N���N=�O�[�N�yvP#�NٗN6�P�N��bO�AO��LN�(�N�	�O�y8N�v�Ol��Pk��N��N��N���N��4N��NwvVN��EN@g�N�-Ne��N��NCC�N�L�O�`WN�G~O��NE(N��O�D��`B��`B�t��o�ě����
��o��o��o��o��o��o%   %   ;o;D��;D��;��
;ě�<o<t�<49X<T��<T��<T��<T��<e`B<e`B<�o<�C�<�C�<�t�<�t�<���<���<���<�1<�1<�1<�1<�j<�j<�j<�j<���<���<�`B<�`B<�`B<�`B<�`B<�h<�h=o=C�=t�=#�
=#�
=8Q�=L��=P�`=T��=aG�=aG�=y�#=y�#=��=�7L=��=��=��=���=�- )5BNPXUMA75)<;A[t����������tgJB<30:<HPTTH<3333333333��������������������^\[anyz}}zxnfa^^^^^^bhist�����thbbbbbbbbbeggt�������}togbbbb``annqqnca``````````-6?BOTUOB6----------��������������������99<@IU_[WUI<99999999�����
2:?></#!
�hm����������������uh	6EQYaa^WK6)	��������������������" )-57@BNXNB@5)""""��������������������#.020,)#nnz{�����{vnnnnnnnnn)5NZdd_aZN5)"#""#&0<IIJIB<700#""������#/BGE8/#
���plpt~�����tpppppppp���	


�����������()6>BO[hopjh][OB65)(~}|���������������~&'(()5BN[ad[W[^[N5*&��������������������/<HaprliaVRH<3#��������������������pmlt��������������wp�)BNW[UOLB85)��#/<CHLMIHA=<8/,+&#"/;HOWYYTLH;/"���������)6BFD5-����������������������")5=BJNU[e[NB5)��������������������34<EHTUV`UH<33333333��������� 
����ggnt~���������|tjgggmt����������������tm���

��������������������������������������
)<U^aU<
����������������������^gq~��������������t^����
#/<HXekYH<,#
��tqz��������ztttttttt�����������)6OV]bZB6)��������������������)/;NTWVKC5) ZYam������������}ykZ?@BDNORQNB??????????�����������������������������������\bmtx�����������yh[\������������������������������������������������������������������������������
)**)









jefnnz����~zpnjjjjjj{�������������������������������������)-1575)toz���������������ztPOQTVYacmopromleaWTP
,5BDMMJFB6)')*6?BCCB6.)''''''''55BN[ee[NB5555555555��������

������6�B�O�V�[�h�k�x�}�w�t�h�O�B�6�)�'�)�4�6�����������������������������ù������D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�D����������������������������������������Ҿ��	������	���������������������L�R�Y�e�f�e�e�Y�L�K�F�J�L�L�L�L�L�L�L�L���������������������������������������������������������������������������������z�|���������z�q�u�x�z�z�z�z�z�z�z�z�z�z�������üʼ̼ʼ�������������������������������������������������m�y���������������y�`�G�:�9�6�;�G�M�T�m�s�����������������������`�N�I�H�S�d�g�s���������ʼۼּ��������r�Y�4�0�7�E�]�r����C�K�O�\�g�\�W�O�C�*��������������������ʾ־׾ؾ׾ʾǾ¾������������������������ĿѿؿտѿĿ������������������������-�:�F�S�W�[�S�P�F�?�:�-�!�"�-�-�-�-�-�-���������������������������������������������	�H�z����o�Z�H�"����������������˼Y�f�r���������������~�r�j�f�Y�Y�X�Y�Y�	��"�/�=�F�E�C�F�/�"��	����������	�\�h�u�y�u�t�h�\�O�O�O�V�\�\�\�\�\�\�\�\�����������������������������������������
������&�&�#��
���������������
��������������������������s�p�k�k������`�m�y�����������}�y�`�X�T�K�G�F�;�G�T�`�)�6�?�B�E�B�6�)��#�)�)�)�)�)�)�)�)�)�)��������"�$������������ùíîù���޺�3�L�X�T�F�@�/���������� �������Ľн��������Ľ��������~������������������ �%������ݿϿĿ����ѿۿ�������������������z�w�m�a�`�a�d�m�z�������T�_�a�`�[�T�H�;�/��	����������	�"�;�T�������û�����������ܻл������d�Q���A�M�S�U�M�A�4�+�4�7�A�A�A�A�A�A�A�A�A�A¦´¹²«�}�{�u�v�y�~�I�V�b�m�o�{Ǆ�{�o�b�a�V�K�I�D�I�I�I�I�I���
��
��
�
�
��������������������������4�A�f�s���������������f�M�4�'����g�g�s�x�������w�s�g�f�Z�N�L�N�S�Z�e�g�g�t�|�{�t�]�V�H�)�����������������B�n�t�����������������������������������������/�<�H�S�L�H�<�/�+�.�/�/�/�/�/�/�/�/�/�/���������þ���׾���������l�^�s�~����ÓØàìùú��ýùìàÓÉÇÅÇÉÈÓÓ�A�F�Z�f�y��������s�f�Z�M�A�9�0�2�4�:�A���5�?�@�M�_�d�d�Z�N�A�5�+����������5�A�J�N�N�N�N�A�5�*�)�/�5�5�5�5�5�5�5�5�.�;�G�J�T�U�V�T�J�G�;�.�"�!�� �"�.�.�.���ʾ׾��	��������׾ʾ������������
��#�/�/�0�/�#����
������� �
�
�
�
�Z�^�]�Z�M�B�A�4�(���� ���(�4�A�M�Z��������������ùàÈÀ�z�{Óì�����=�I�T�V�[�V�I�=�3�3�=�=�=�=�=�=�=�=�=�=��"�.�;�D�;�;�/�.�"���	�����	�����������������������������������������ػ-�:�F�P�S�X�S�P�F�B�:�-�!������ �-�/�<�=�A�@�<�/�#�����#�'�/�/�/�/�/�/�y�������������y�m�i�m�q�y�y�y�y�y�y�y�y���������������������������y�x�l�a�l�y����'�4�5�>�4�'���������������������������������������������������������������������������'�4�@�B�M�M�C�@�4�'����	�����$�'�����������������������<�I�U�b�e�n�o�n�g�b�U�O�I�B�C�>�<�9�<�<�����ͼͼм׼�����ּʼ����������������n�{ŇőŔŠŷŹ��ŹŭŠŔŊŇ�{�t�n�l�n�e�r���������ɺֺӺɺ������~�r�T�F�J�]�e��������������������������������ĽĽ˽ϽͽĽ������������ĽĽĽĽĽĽĽ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�DtDoDrD{D�D� 9 U < U / W 0 h � A D F 6 4 n R [ A F S F T @ X t L 8 3 # Z W < A G S 6 J N s \ 1 s \ > b 0 H h 6   0 / = , Y J / U ) N F K S J [ L 4 Q x P 6 S     �  �  �  N  �  i  �      w  �  �    �  �  �  �  �  N  +  �  �  ^  r  j  Q  �  C  �  Z  �  �  :  �  F  8  �  �  �  �  �  �  S  ?  2    �  k  �  �  �  �  �  �  2    �  >  �  r  �  h  Z  �    s  �  �  P  �  P  �  ��u�D�����
%   ;D��;ě�;��
�o�o;�o;D��<���<�h>;dZ<��
<#�
<D��<u<t�=ě�<�o<�`B<�1<�o<���=o<�h<�`B=��w=�o=q��=8Q�<���=�C�>O�<�9X=o<���<���=P�`<���=��<�`B=o=ix�=<j=<j=Y�=+=#�
=u=�w='�>�=�P=@�=u=@�=]/=e`B=m�h=u=u=�%=�{=��=�hs>�+=��
>V=���=�{>�7LBa�B	]�B�B��B��B2MB	�EB��B'OB T	B&�#BDgB�B��B B�B��B%�B(܉BDB%��B�rB
�Br�B��B��B�.B��B�B_%BmWBԣBu�A�[BUhB��Br�B.�B)�B#s B
�B�FB3|B��BB�B!��BLwB�.B��B�[B�B�UBBwuB�B!oVB� BxUB�B %�B,�SB�IB$;B~BA2B!`�BIdBA�E7BR�B�/B!xBMB^�B��B!�B�=B��B@�B	�]B��B�\B E�B&��B@6B/�BEiBA&B�BC�B%?�B(��B<wB&+�B1cB
>ZB�nBBBRLB�	B�SB��B?B^B��B�A��!BA�Bc"BsB�B1�B#?�B

	B��BD'B�B�:B"<B��B8B��BHtB�@B��BHB|lB)�B!H�B�B��B�B % B,L�BB�B?�B�B>-B!��B�B�|A��tB?XB��B@�B9�A��?A��RC�V�A�Q�AZd	?�xA�#�A�stA��U@�U�A.�aAk"A�(�@��A���AOq�Av�g@HW@�A���@�DwA�J�BxA�3�A�T�AGEYAi�JA�*�A�a?���A(^�A���A�uHA�h�@��OA;P�A���B(�A��iA?;A��MA�;�A��2Aø"AILA˜A?LwA��-A��Ab�5AS_AA��UA9��A��B!MA^��A吞@x'�A�<hAn��A(<@ɿ�A��W@�f3@�M�@��JA�1:@��[A�@
��@Ry�A&1C�ؐAڂ
A�gC�^3AЇ�AZ?��aA�u�A��lA��.@��A/��AkA��A@�XA��OAO�Au�@��@�A��5@��\A�u>B�eA�W-A�z�AG�Ai A�o�A�G?��A(�5A�uA��(A���@�%[A;9�A��0B�A�v�A;5�A���A�k�A���A�|IAM8�A�K�A?�A�G�A��9Ac AR�
A�}�A8�8A���B'�A]�A�Y�@tfA�p
Ao��A p@��A�~�@��$@��@��A��@��(A�@Л@T�A%�C�ל            	   
      
         	            �         
         Y         	                  ?   0   +         2   x                     H      
   !                  !         z               
         	      	            Q      C      
   �      #                                 +   -                  1                           !   %   #   #      !   9               !      /         -      !   '                  /                                                #                                                +                                                                                 !               -         '                  %                                                         ObFTOx�Nk
NeN�xN%��Nj��M���N�N3}N�h�OS��P&kO�LN���NFe�N��N���N"��O^�4N�ɛO�?�NS&/N=�NN��]N���OKN3�N�O�OM� O�$�O;�O0�5Of�N!�O3N���N=�O�[�N�yvOt�NٗN6�P�N��bO\@�O�+N�(�N�	�O�6�N�v�Ol��O�  N��N��N���NѮ-N��NwvVN��EN@g�N�-Ne��N7qN,n�N�L�O��EN�G~Og�NE(N��O=DN  �  �  �    A  "  w  �  h  �    �      �  �    �  �  	1  O  �  �  �  �  �  �  k    j  <  c  b  �  
b  Y  �  �  g  \  �  �  �  �  �  W  �  q  "  F  l  �  �  �  �  �  *  q  �    9  4  �  <  �  0    �  D  	�  �    
��`B���ͼt��o���
���
�o��o��o�D����o;ě�;o=��w<t�;��
;D��;��
;ě�=aG�<#�
<T��<T��<T��<T��<u<u<e`B=P�`=t�<��<��
<�t�=\)=�1<���<�1<�1<�1<�1<�j=T��<�j<�j<���<���=+<�h<�`B<�`B<��<�h<�h=�t�=C�=t�=#�
='�=8Q�=L��=P�`=T��=aG�=aG�=�O�=}�=��=�\)=��=�Q�=��=���>�� )5BNPXUMA75)GCBDN[t�������tjg[NG30:<HPTTH<3333333333��������������������_]^anwz||zunia______bhist�����thbbbbbbbbhgoty������thhhhhhhh``annqqnca``````````-6?BOTUOB6----------��������������������99<@IU_[WUI<99999999����
#/6782/'# 
�ko����������������zk)6BHLONLHB6)��������������������' $)5<BDB50)''''''''��������������������#.020,)#nnz{�����{vnnnnnnnnn$)5BIMLJDB5)'$$)0<EG@<40''''''''������
/<C>/#
����plpt~�����tpppppppp���	


�����������()6>BO[hopjh][OB65)(~~���������������'()*5BN[_ab[V[NB5+)'��������������������-,/0<HQUWUQHA<6/----��������������������tu��������������yvt��)BNSXWTNJB5)!�#/<CHLMIHA=<8/,+&#(%##&/;HMRROIH?;//((��������������������������������������")5=BJNU[e[NB5)��������������������34<EHTUV`UH<33333333��������� 
����ggnt~���������|tjggg�����������������������

��������������������������������������
)<U^aU<
����������������������|xz����������������|����
#/<HV`eVH<4#
��tqz��������ztttttttt�����������
	)6OZ^WOB6)
��������������������)/;NTWVKC5) z~����������������{z?@BDNORQNB??????????�����������������������������������e]cntz�������~��|the������������������������������������������������������������������������������
)**)









jefnnz����~zpnjjjjjj~�����������~~~~~~~~��������������������)-1575)usz���������������zuPOQTVYacmopromleaWTP)6BDAA=76/)')*6?BCCB6.)''''''''55BN[ee[NB5555555555��������

������6�B�O�V�[�h�k�x�}�w�t�h�O�B�6�)�'�)�4�6��������� �� ��������������������������D�EEEEEED�D�D�D�D�D�D�D�D�D�D�D�D����������������������������������������Ҿ��	������	����������������������L�R�Y�e�f�e�e�Y�L�K�F�J�L�L�L�L�L�L�L�L���������������������������������������������������������������������������������z�|���������z�q�u�x�z�z�z�z�z�z�z�z�z�z�������¼ļ�����������������������������������������������������m�y�������������y�m�`�T�Q�G�E�G�L�T�`�m�s�����������������������g�N�K�U�Z�e�g�s����������������������r�f�\�T�W�c�r�������*�6�6�6�*� �����������������������ʾѾԾʾ��������������������������������ĿѿؿտѿĿ������������������������-�:�F�S�W�[�S�P�F�?�:�-�!�"�-�-�-�-�-�-�������������������������������������������	��"�/�;�B�D�;�/�"��	���������������f�r�����������r�l�f�^�f�f�f�f�f�f�f�f�	�"�/�;�@�C�C�>�;�0�"����	���������	�\�h�u�y�u�t�h�\�O�O�O�V�\�\�\�\�\�\�\�\�����������������������������������������
������&�&�#��
���������������
����������������������s�r�m�n�s��������`�m�y��������|�y�m�`�Z�T�G�<�B�H�T�[�`�)�6�?�B�E�B�6�)��#�)�)�)�)�)�)�)�)�)�)�����������������������������������'�(�3�<�@�@�@�>�5�3�'����
�����'�ݽ���������ڽнĽ��������������Ľнݿ�������!�������ݿҿſſѿ޿��������������������z�w�m�a�`�a�d�m�z��������"�/�;�H�Q�S�N�H�;�/�"���	��	�	���������ûǻлԻӻлͻû������������������A�M�S�U�M�A�4�+�4�7�A�A�A�A�A�A�A�A�A�A¦´¹²«�}�{�u�v�y�~�I�V�b�m�o�{Ǆ�{�o�b�a�V�K�I�D�I�I�I�I�I���
��
��
�
�
��������������������������4�A�f�s���������������f�M�4�'����g�g�s�x�������w�s�g�f�Z�N�L�N�S�Z�e�g�g���)�5�B�C�D�B�7�)��������������������������������������������������������/�<�H�S�L�H�<�/�+�.�/�/�/�/�/�/�/�/�/�/���������þ���׾���������l�^�s�~����ÓØàìùú��ýùìàÓÉÇÅÇÉÈÓÓ�A�M�Z�a�s�~����}�s�f�Z�M�B�A�8�7�8�?�A���5�>�?�K�]�b�c�Z�N�A�5�-����������5�A�J�N�N�N�N�A�5�*�)�/�5�5�5�5�5�5�5�5�.�;�G�J�T�U�V�T�J�G�;�.�"�!�� �"�.�.�.���ʾ׾������ ����׾ʾľ������������
��#�/�/�0�/�#����
������� �
�
�
�
�Z�^�]�Z�M�B�A�4�(���� ���(�4�A�M�Z�����������������ùìÞÖÒÑ×ê�����=�I�T�V�[�V�I�=�3�3�=�=�=�=�=�=�=�=�=�=��"�.�;�D�;�;�/�.�"���	�����	�����������������������������������������ػ!�-�:�F�N�S�W�S�M�F�C�:�0�-����� �!�/�<�=�A�@�<�/�#�����#�'�/�/�/�/�/�/�y�������������y�m�i�m�q�y�y�y�y�y�y�y�y���������������������������y�x�l�a�l�y����'�4�5�>�4�'���������������������������������������������������������������������������4�;�@�H�@�6�4�'�!��'�-�4�4�4�4�4�4�4�4�����������������������<�I�U�b�e�n�o�n�g�b�U�O�I�B�C�>�<�9�<�<�����˼ϼּ����ּʼ������������������n�{ŇőŔŠŷŹ��ŹŭŠŔŊŇ�{�t�n�l�n�r�~�������������������~�r�e�c�Y�R�Y�e�r��������������������������������ĽĽ˽ϽͽĽ������������ĽĽĽĽĽĽĽ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{D~D�D�D� 9 F < U + W ? h � 7 D 3 6 $ Y / [ A F 5 5 X @ X t B + 3  & E 6 A 2  6 J N s \ 1 > \ > b 0 , g 6   - / = ! Y J / S ) N F K S J 1 E 4 O x D 6 S     �    �  N  ~  i  �      J  �  �  �    '  g  �  �  N  �  �  �  ^  r  j    �  C  �  P  �  �  :  |  �  8  �  �  �  �  �     S  ?  2    �  F  �  �  H  �  �  8  2    �    �  r  �  h  Z  �  I  T  �  �  P  �  P  �  �  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  ?:  �  �  �  �  �  �  �  �  �  �  i  S  :        �  �  �  _  �  �  �  �  �  �  �  �  �  �  �  �  y  l  W  8    �  n   �  �  �  �  �  �  u  h  Z  M  ?  2  &        �  �  �  �  �      '  %      �  �  �  �  �  {  [  9    �  �  �  �  _  9  =  A  =  9  7  6  *      �  �  �  �  `  ;    �  �  �  "  6  D  C  >  6  )      �  �  �  ]  !  �  �  c     �  �  F  R  _  i  t  |  �  �  z  r  i  _  T  6  �  �  �  r  N  *  �  �  �             
      �  �  �  �  �  �  �  �  s  h  l  p  t  x  |  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  Y  ;    �  �      �  �  �  �  �  �  �    l  X  =  #  	  �  �  �  �  �    F  r  �  �  �  �  �  �  m  N  #  �  �  K  �  �  ;  �   �         �  �  �  �    c  L  @  4       �  �  |  #  �  =  �  �  �  �  r    �  �    �  �  E  �    "  �    h  	�  �    M  g  w    �  �  �  �  x  `  E  *    �  �  �  �  �  2  p  u  z  �  �  �  �  �  �  �  �  �  �  �  �  g  J  2            �  �  �  �  �  �    d  G  '  
  �  �  �  �  X  )  �  �  �  �  �  �  �  �  �  �  �  h  B    �  �  w  ?     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    x  q  i  �  �      �  �  �  	  	  	%  	1  	.  	  �  �    `  9  �  �  3  =  H  N  J  E  ?  7  /  $    
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  x  �  �  �  w  ]  9    �  �  �  �  �  �    p  `  P  ?  /      �  �  �  �  �  �  �  �  }  r  �  �  �  �  �  �  �  �  �  �  �  }  s  f  X  I  :  +      �  �  �  �  �  �  �  �  �  �  �  �  t  i  c  ]  T  ;  !    �  �  �  �  �  �  �  �  u  T  /    �  �  a  '  �  �    �  �  �  �  �  �  �  �  k  R  5    �  �  �  �  `  5  �  @   �  k  |  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  v  �  �  0  j  �  �  >  {  �  �  �      �  �  N  �  G  �  �  �  �  �  �  H  �  �  (  Z  i  Y  9    �  �  (  �  Y  �  F  �    ^  �  �  �  !  8  ;  3  !    �  �  d    �    �  �  _  ^  c  `  X  R  J  B  ;  4  *      �  �  �  L  �  �    �  b  Y  P  G  <  2  )         �  �  �  �  �  �  �    k  V  �  ,  W  �  �  �  �  �  �  �  �  �  m  &  �  D  \  2  �  4  �  	H  	q  	�  	�  	�  	�  	�  
G  
O  
]  
Q  
  	�  	;  �  �  �  �  �  Y  U  P  K  F  A  =  7  2  ,  '  !        �  �  �  �  �  �  �  �  �  �  n  [  G  .      �  �  �  �  �  �  �  y  -  �  �  �  {  l  \  L  <  *      �  �  �  �  �  �  }  r  g  g  Z  L  ?  2  %        �  �  �  �  �  �  �  �  �  \  9  \  Z  X  Q  G  :  +       �  �  �  k  *  �  �  Y  �  �  
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  p  d  Y  M  B    4  �    q  �  �  �  �  �  �  v  *  �  D  �      �  T  �  �  �  �  �  �  �  �  �  ~  `  A    �  �  �  �  \  3    �  �  �  �  �  �  �  �  u  e  Q  8    �  �  �  �  s  N  )  �  �  �  �  �  �  �  �  ~  �  �  ~  k  8    �  �    �    W  I  -    �  �  �  {  R  $  �  �  �  w  C    "  B  f  �  x  �  �  �  �  �  �  �  �  �  o  M  '  �  �  �  S        f  p  i  [  J  2    �  �  �  �  �  l  w  A    �  x  /  �  "    
  �  �  �  �  �  �  �  �  �  �  �  z  i  b  \  W  R  F  B  ;  3  $    �  �  �  �  �  �  s  U  1    �  �  d    V  i  l  j  f  ]  S  I  A  9  .    �  �  �  Y    �    {  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  q  `  M  8    �  �  �  �  �  |  d  V  A  *      �  	�  
�  A  �  B  �  �  �  �  r  ,  �  j  
�  
I  	m  W  -  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  g  O  5    �  �  �  �  x  R  *    �  �  �  *  )  &      �  �  �  �  Q    �  �  1  �  �  5  �  ^  �  \  d  k  n  e  [  Z  ^  b  E  #    �  �  ~  C    �  �  �  �  �  �  �  v  a  M  7       �  �  �  �  �  �  �  �  �        �  �  �  �  �  �  �  �  �    q  c  W  J  >  0  #    9  4  /  )          	    �  �  �  �  �  �  �  �  �  �  4  !    �  �  �  �  �  �  �    o  a  c  d  s  �  �  �  �  �  �  �  �  �  �  �  z  �  �  �  �        &  +  0  5  :  <  -      �  �  �  �  �  �  |  ]  ?  %  
  �  �  �  �  �  �  �  r  [  g  �  �  �  �  �  �  ]  (  �  �  g    �  z  -  %  ,  )      �  �  �  �  �  m  K  <  0  '           �    �  �  �  �  �  �  �  �  �  �  ~  j  Z  P  G  <  .       �  �  �  h  �  �  �  �  {  3  �  �  C  
�  
  	<    �  �    D  /      �  �  �  �  �  �  �  �  V  '    �  �  �  �    	,  	�  	�  	�  	�  	�  	�  	o  	G  	  �  �  M  �  �  �  �  �  9  �  �  �  �  �  �    [  0    �  �  r  A    �  �  �  W  '  �      �  �  �  �  �  �  i  Q  8    �  �  �  �  <  �     X  �  C  �  %  �  �  �    �  �  �  #  l  P  �  \  �  �  $  
@