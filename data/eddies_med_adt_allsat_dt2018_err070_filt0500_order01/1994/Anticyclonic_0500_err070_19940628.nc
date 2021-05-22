CDF       
      obs    8   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��`A�7L      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N w�   max       Px��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �D��   max       =���      �  l   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @D��
=p�     �   L   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ə����    max       @v�33334     �  )   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @Q            p  1�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @�=           �  2<   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �,1   max       >1&�      �  3   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B$�      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B$�]      �  4�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C��      �  5�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�6{   max       C���      �  6�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7|   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          3      �  8\   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          -      �  9<   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N w�   max       P,�I      �  :   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�MjOw   max       ?��N;�5�      �  :�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �D��   max       =���      �  ;�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @D��Q�     �  <�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�33334    max       @v�fffff     �  E|   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @$         max       @Q            p  N<   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @�@          �  N�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?y   max         ?y      �  O�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���%��2   max       ?��c�A \     �  Pl      	               *         ,   0   
   2                5      !               D      $                              G         W   	                  �               "      !   /   5NaFWN��5N`sjN�9�N��N�A�O���N<��N`�O>d:P:@�O�O�.N�IN��LNנ�P,�IP.��NE�O�.�N���O��>N.�OV�O�$,Ny�O��_N_�O���O[�N1>N&��N�-5N+��N w�NO)�Px��P	�N�t$PP�"N%J�PQ�N3�BO���N\��Ob	{O�ICO\:N�v�O�#2OF�(O5�OOKO��O�-O��!�D����j��o�#�
�#�
��o�o:�o:�o;�o;�o;ě�;ě�;ě�;ě�<o<t�<T��<T��<u<�o<�o<�o<�t�<�t�<�1<�9X<�9X<�9X<�j<�j<�j<�j<ě�<���<�`B<�h<�<�<��<��=o=C�=t�=�P=#�
=,1=0 �=49X=<j=L��=P�`=�C�=��=��=������������������������������������������������������������NNYUZ[giooog[NNNNNNN���������	�������������������������������
#03:=?;5/#
�����

��������������������������������SOUanz��������}zsn[S23>F[����������hOB72��������	������ '/<HUnz��wjaUH</$ st������tssssssssss��������������������llpt�����������~wtll������5BGPYPB5���y|����������������y<<?BGLNOPPNONB<<<<<<���������������������������������������������

�������)-40)#"(/;HLOPNH;/-"��������������������	
#/91/#
								��������������������.+)&,./4<=><<30/....)*69=HHB6����������������������������������)68:6)��������������������/&+/5<HTH<0/////////(),*)(lqt����������tllllll!)5B[g��������[B.!HNUalz���������zaWNH223166BO[]_[VOKB?622������������������������������������6BNgr���������g^^N56�	got�������������yrgg��������������������)5BNPSRRPJB5)'��������

�����	
#+0940.+#
	st����������������ts��������
���������)-.+)(!�vvz����������������v76<FHTadgihfcaZTHB;7 )26ABKO[bgj_OB6)�)6B[hmqn_6���������
%&
�����ɼ�����
�
���������߼������čĚĠĦĳĽĳıĦĚĒčĈĉčččččč�������������������������������O�O�[�h�tĀ�{�t�h�[�O�K�O�O�O�O�O�O�O�O�������������������������������������������	�����	��������������������������m�y���������ƿƿ��������y�`�T�L�E�I�U�m�B�O�S�[�O�G�B�=�7�<�B�B�B�B�B�B�B�B�B�B����������������������������������������FJFVF^FdFdFaFVFMF=F$FFF	FFFF$F(F=FJ�A�M�Z�f�w������f�M�A�(���	� ���(�A���!�(�5�8�A�G�H�A�5�(����
����
�����$�'�'�"��	��������������������àæåçàÓÇÊÓßàààààààààà�`�k�m�y���������y�m�`�T�N�T�V�[�`�`�`�`�����ĽɽʽɽĽ��������������������������O�\�hƎƖƜƤƫƿƹƧƚ�u�\�M�I�E�>�C�O�����������������Z�A�(����*�A�N�g�����Ŀѿݿ���������ݿѿĿÿ��ĿĿĿĿĿ�ù����������
���������ùìàË×àù���������������������{�s�j�s�v�}���������A�N�Z�Y�U�I�A�5�#������������*�A��������������������������������������������"�/�;�G�;�7�2�$��	������������������!� �.�'������ܹϹ����������Ϲ��������"����
����������àçêèÛÓÐÇ�z�a�S�X�e�^�a�`�n�zÇà�m�z�����������������������z�m�h�m�m�m�m�����׾����"�7�1�.�'�"��׾ʾ�������������������������������������������������ù������������ùí÷ùùùùùùùùùùÇÉÓÔÕÕÓÇÂÂÀ�ÇÇÇÇÇÇÇÇùù����������ùìàÓÐÌÓÖßàìøù�����
��
���������������������������񿒿������������������������������������������������ �������������������������������������������s�g�P�J�e�}�����5�Z�r�����s�q���������Z����������5�лܻ�����
���������ܻѻлȻл��U�a�k�q�|ÃÀ�{�p�H�/�#�����������;�UE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¥�t�g�[�5�)����(�B�N�g���������������������������������������������������������������y�s�m�`�Z�`�m�v�������������������������������������������0�:�>�H�K�P�K�I�=�0�$�����	���.�0D�D�D�D�D�D�D�D�D�D�D�D�D�D�DnDgDiDoD{D����������!�%�!��������ټټ������������������������������������������׾��	���� ������ʾ����������Ͼ���#�0�<�I�T�V�U�F�<�0�#���
����
���������(�)�������������������������#�0�<�?�I�D�<�0�#��
�������������
��~�������ƺֺߺֺԺǺ����������~�o�e�o�~�r�t����������ɼȼü����������r�p�j�o�rEPEiE�E�E�E�E�E�E�E�E�E�EuErEpEiEPE@EBEP 2 C 8 ^ x _ A D p d / V : a E % = R } Q N \ \ A > � \ | f P h ~ G w \ t H ; I F 2 V O M < 0  0 f 2 2 & ; U � b  x  �  d  �    �  �  o  �  �  E  K  �  Q  �  �    ^  �  �  �  U  7  �  T  �  �  �    h  9  �  )  k  G  �  (  �    �  9  �  @  K  y  �  |  R  �  �  �  �  �  g  N  ��,1�e`B�t���`B�ě�<49X=�w;�`B<o=D��=P�`<�o=e`B<t�<D��<e`B=#�
=�+<�C�=D��<ě�=C�<�9X=+=� �<�j=m�h<�/=8Q�=<j<�h<�`B=T��<�h<�`B=o=���=m�h=8Q�==�w=m�h=�P=P�`=#�
=u>1&�=e`B=P�`=�C�=�o=�{=ě�=�
=>�u> ĜB�B�VB�B��B�B6�B�$B-B"]B��Bd8B�PBwB�KB H�BҧB�B`B�jBB�B��B�B7	A��B�B��B�B4jBkBf�B�B�B!�/B��B��B� B	C�BfHB]Bh�B�DB	��B��B
�CBegB<XB�B$�Bf�B#6�B�pB
��A�P#B.B&�B��B��B�B��B��B4�B?�B��B>�B"AiB�B�nB��B1�B�XB L�B��BB�B��B@�B>dBAiB@$A���B�<B7BòB@B=�B��B�WB�B"> B��BBBdB	�B�mBD2Ba�B�QB
@$B��B:6BP�B?�B6�B$�]B�vB"�B4TB
�jA�~B>"BKB�8A�RA��H?L�
A���A���A��}Al��A��G@D C��A<.A�5A��lA�bAk2�A#�B�	A�ddA|w[A���A� �A��A��4A�r�?��A3{�AȾ�A��AU�A�C�A�7�A�lHA�X(A���Aq@�@��A��(A�Fy@�gA�UC�arA���A�|gApb�A�TB

AC��UAl�A�B�AU"�A�.A�P�A�2@A#@�RjC���A�A߉(?P�AۃEA�(A�{DAl�#A�}�@ҕC���A<�HA��A�}eA���Ak'xA$��B��A���Az��Aς�A�y*A��A�xdA��>�6{A4�AȄ�A�~�AR�VAЀ$A�9�A�|~A�q�A���Ap�@�*�A�yZA��a@���A�:C�`�A�t�A�~�As'A�
B	��C���A.;A�dvAV;A��Aӈ�A�<r@B@�/C��      
               *         -   0      3                6      "   	            E      %                              H         W   	                  �               #      !   /   6                     #            /                  )   +      !               '      #      %                        3   -      1      +                        !            #   !   '                                 !                  )   #                                 !                        '   -      #      +                        !            !      'NaFWNXSlN`sjN�9�N��N��O"[5N<��N`�N��O��vN�]�NᤞN�INq��Nנ�P,�IO�NE�N�k�NWa�O��>N.�OC��OƉNy�N0J=N_�O�O[�N1>N&��N��:N+��N w�NO)�P�P	�N�t$O��N%J�PQ�N3�BO���N\��OL�vO#`�O��N�v�O�#2O0�LO&	.O]{O��O��UO��!  �  `    �  4  �  v  >  a  
  7  �    u  >  �  �    �  9  �  �  �  �  K  �  {  O  '  	  �  �  �  �  y  �    �  e  	�    ]  �  �  �  �  3  K  �  ;  �  ^    ,  
�  	��D����1��o�#�
�#�
�D��<e`B:�o:�o<e`B<�C�;�`B<��;ě�;�`B<o<t�<�/<T��=+<�t�<�o<�o<���=e`B<�1=0 �<�9X<�j<�j<�j<�j<ě�<ě�<���<�`B=Y�<�<�=m�h<��=o=C�=t�=�P='�=��=49X=49X=<j=P�`=Y�=��=���=���=������������������������������������������������������������NNYUZ[giooog[NNNNNNN���������	����������������������������
#,/5750/$#
����

��������������������������������^WUanz������zxnja^^JEGPft���������th[OJ��������������////<HLU\_YUH<0/////st������tssssssssss��������������������llpt�����������~wtll������5BGPYPB5�����������������������<<?BGLNOPPNONB<<<<<<����������������������������������������������

�������)-40)#"&/;HKNOLH;/"��������������������	
#/91/#
								��������������������.+)&,./4<=><<30/....
)68<HGB6����������������������������������)68:6)��������������������/&+/5<HTH<0/////////(),*)(lqt����������tllllll/.5B[t���������g`B5/HNUalz���������zaWNH223166BO[]_[VOKB?622��������������������������������������6BNgr���������g^^N56�	got�������������yrgg��������������������)5BOQQQNHB5)("�������

 �����
!#&0520/-*#
st����������������ts��������
�������� %),-)&	�ww{����������������w::;@HTTacefdaaTHE>;:!')36BOU[`eh\OB6)	)/6BO[hlqm_D6��������
%&
�����ɼ�����
�
���������߼������čĚĦĳĹĳĬĦĚĘčČčččččččč�������������������������������O�O�[�h�tĀ�{�t�h�[�O�K�O�O�O�O�O�O�O�O�������������������������������������������	�����	��������������������������m�y�����������������y�m�`�_�T�R�U�`�e�m�B�O�S�[�O�G�B�=�7�<�B�B�B�B�B�B�B�B�B�B����������������������������������������F=FJFVF_F_F[FVFJF>F=F1F)F$F!F$F,F1F:F=F=�4�A�M�Z�s�x��~�s�k�M�A�4�$�����-�4����(�5�5�A�E�E�A�5�(�����������������	�����������������������àæåçàÓÇÊÓßàààààààààà�`�h�m�y���������y�m�`�W�X�]�`�`�`�`�`�`�����ĽɽʽɽĽ��������������������������O�\�hƎƖƜƤƫƿƹƧƚ�u�\�M�I�E�>�C�O�g�������������������s�Z�A�5�(�.�6�A�N�g�Ŀѿݿ���������ݿѿĿÿ��ĿĿĿĿĿ�����������������������������ùïùü�����������������������v�y�������������������A�N�Z�Y�U�I�A�5�#������������*�A�������������������������������������������	��"�/�;�B�;�5�0�"��	���������������ܹ������������ܹҹϹù��ùĹϹܾ�����"����
�����������zÇÎÈÇ�}�z�w�n�l�n�q�z�z�z�z�z�z�z�z�m�z�����������������������z�m�h�m�m�m�m�����ʾ׾����"�0�-�'�"��׾ʾ�������������������������������������������������ù������������ùí÷ùùùùùùùùùùÇÉÓÔÕÕÓÇÂÂÀ�ÇÇÇÇÇÇÇÇàìù����������ùìëàÓÐÌÓÖßàà�����
��
���������������������������񿒿������������������������������������������������ �����������������������������������������r�g�b�`�m�s�������5�Z�r�����s�q���������Z����������5�лܻ�����
���������ܻѻлȻл��H�U�a�q�t�s�n�c�U�H�/�#�
� ����(�<�HE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�¥�t�g�[�5�)����(�B�N�g���������������������������������������������������������������y�s�m�`�Z�`�m�v�������������������������������������������3�;�F�I�O�I�=�0�$�����
��� �$�/�3D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DyD|D�D�D���������!�"�!���������ڼڼ������������������������������������������׾��	���� ������ʾ����������Ͼ���#�0�<�G�I�Q�T�I�D�<�0���
�	��
���������&�'������������������������#�&�0�9�<�<�6�0�#��
�����������
���~���������úֺкɺź������������p�g�q�~�r������������ɼȼǼü������������r�k�rEPEiE�E�E�E�E�E�E�E�E�E�EuErEpEiEPE@EBEP 2 I 8 ^ x Z  D p J % T , a @ % = O } U I \ \ @ 4 � H | c P h ~ D w \ t D ; I @ 2 V O M < ,  1 f 2 - " 3 Y � b  x  �  d  �    �  V  o  �    �  (  �  Q  �  �    7  �    �  U  7  �    �  Q  �  �  h  9  �  �  k  G  �  �  �    '  9  �  @  K  y  �  W  /  �  �  y  `  L  1  �  �  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  ?y  �  �  �  �  �  �  �  �  �  �  u  e  U  E  4  #          Z  \  _  `  `  U  F  ;  0  #      �  �  �  �  �  �  �  k    �  �  �  �  �  �  �  �  �  �  �  |  s  h  [  N  [  q  �  �  �  �  �  �  �  �  �  �  �  {  k  [  M  A  6  +      	  4  +  #      	    �  �  �  �  �  �  �  �  �  �  �  |  i  �  �  �  �  �  �  �  m  G    �  �  z  ;  �  �  �  O    �  �  �    #  @  _  o  v  s  f  I    �  �  I  �  9  r  �  �  >  /         �  �  �  �  �  �  �  �  �  M  �  �  �  r  P  a  V  L  >  '    �  �  �  �  �  b  @    �  �  �  �  �  v  	�  	�  	�  	�  	�  	�  	�  	�  	{  	>  �  �  H  �  <  �  �  �  g  �  �       3  5  7  6  '    �  �  �  �  �  �  ]  *  �  �    �  �  �  �  �  �  �  }  h  Q  9       �  �  �  Q     �   �  �  �  G  �  �  �  �  �        	  �  �  �  $  �  7  �    u  z    �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  u  n  <  <  =  >  =  <  ;  5  ,  $      �  �  �  �  �  �  m  R  �  �  �  �  �  �  �    g  P  :  $     �   �   �   �      
  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  P    �  ;  �  E  �  �  �        �  �  �  �  �  i  )  �  N  �  �      �  �  �  �  t  g  W  F  6  &    �  �  �  �  �  x  [  >  "  �  ,  _  �  �  �  �      -  7  3    �  X  �  e  �  M  �  p  w  ~  �  �  �  �  �  x  p  g  ]  T  H  ;  4  -      �  �  �  {  i  S  6    �  �  �  �  �  �  W     �  �  f    �  �  �  �  y  m  a  U  I  ?  4  !    �  �  �  s  H     �   �  �  �  �  �  �  �  �  �  �  �  �  k  L  %  �  �  {  ;  �  �  O  �  �  1  �  �  �    /  7  G  H  )  �  �  5  �    j  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ^  [  F  =  r  �  A  q  �  �  �  d  p  W    �  _      �  O  H  B  ;  4  ,  #      �  �  �  �  �  �  �  w  n  e  \    %          �  �  �  �  �  w  a  A    �  �  =  �  N  	     �  �  �  �  �  �  �  �  �  �  �  p  B    �  �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  C  �  �  �  �  �  s  d  U  F  1    �  �  �    S  '  �  l  
   �  B  �  �  �  �  �  �  _  6    �  �  �  J  �  �  3  �  �  &  �  �  �  �  �  �  ~  j  T  <  %    �  �  �  �  �  �  h  J  y  o  e  [  Q  G  =  4  *              �   �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      
  	  �  �  �  �  3  �  :  �  d    3  8  �  z  x  �  w  ^  <    �  �  �  P    �  �  �  T    �  u  e  _  Y  U  W  \  X  M  :  "     �  �  Z    �  0  �  V   �    m  �  	  	X  	z  	�  	w  	P  	  �  h    �  �    l  �  �  B      �  �  �  �  �  �  |  f  N  2    �  �  �  �  �  w  X  ]  O  B  6  .  (    �  �  �  �  T    �  �  �  b  C  �  +  �  �  �  �  �  �  �  �  �  �  �  �  ~  t  t  s  r  r  q  p  �  �  �  �  �  |  Q  $  �  �  �  �  i  U  N  1  �  �  q  +  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  q  e  �  �  �  �  �  �  �  �  �  �  �  �  �  i  >    �  �  m  N  m  A  �  W  �  �    0  -    �  U  �  �    -  3  �  
P  8  J  K  I  E  ?  8  0  $      �  �  �  �  q  A  
  �  i    �  �  �  �  �  z  i  `  W  +  �  �  �  �  }  c  H  8  ,     ;  9  8  .      �  �  �  �  �  u  T  )  �  �  e    �  �  �  �  �  �  �  r  ^  I  0  "      �  �  �  �  �  �  �    L  ]  Q  2    �  �  �  M    �  �  B  �  �    �  '  f  +  �  �  �     �  �  �  �  �  T    �  �  ^    �  �  z  L  �    )    �  �  �  d  8    �  �  X    �  q    �  !  �  +  
d  
�  
~  
^  
I  	�  	�  	D  	  �  �    �    �  1  �  �  n  �  	�  	c  	  �  �  	w  	�  	�  	`  	;  	D  	  �  �    ~  �    $  �